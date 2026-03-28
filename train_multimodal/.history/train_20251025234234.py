from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
import zipfile
from PIL import Image
import io
import os
import json
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from typing import List, Dict, Any

# 检查CUDA是否可用
assert torch.cuda.is_available(), "CUDA不可用，请检查GPU驱动和PyTorch安装"
device = torch.device("cuda")  # 显式定义设备为GPU
print(device)

# 定义所有本地模型路径（请根据实际下载路径修改）
LOCAL_LLM_PATH = '/root/autodl-tmp/llm_related/train_multimodal_from_scratch/Qwen2.5-0.5B-Instruct'
# 第一个SigLIP模型：so400m版本
LOCAL_VISION_MODEL_SO400M = '/root/autodl-tmp/llm_related/train_multimodal_from_scratch/siglip-so400m-patch14-384'
# 第二个SigLIP模型：base版本（新增变量）
LOCAL_VISION_MODEL_BASE = '/root/autodl-tmp/llm_related/train_multimodal_from_scratch/siglip-base-patch16-224'

class VLMConfig(PretrainedConfig):
    model_type = "vlm_model"
    def __init__(self,
                 llm_model_path = LOCAL_LLM_PATH,
                 vision_model_path = LOCAL_VISION_MODEL_SO400M,
                 # 冻结视觉模型和文本模型的参数，仅训练图像向文本对齐的那两个全连接层的权重
                 # 也就是视觉模型用来对齐的那两个全连接层
                 freeze_vision_model = True,
                 image_pad_num = 49,
                **kwargs):
        self.vision_model_path = vision_model_path
        self.llm_model_path = llm_model_path
        self.freeze_vision_model = freeze_vision_model
        self.image_pad_num = image_pad_num
        super().__init__(**kwargs)
        
        
        
class VLM(PreTrainedModel):
    config_class = VLMConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # 1. 加载模型时不立即转移设备
        self.vision_model = AutoModel.from_pretrained(self.config.vision_model_path, local_files_only=True)
        self.processor = AutoProcessor.from_pretrained(self.config.vision_model_path, local_files_only=True)
        self.llm_model = AutoModelForCausalLM.from_pretrained(self.config.llm_model_path, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_path, local_files_only=True)

        # 2. 线性层也不指定设备
        self.linear1 = nn.Linear(
            self.vision_model.config.vision_config.hidden_size * 4, 
            self.llm_model.config.hidden_size
        )
        self.linear2 = nn.Linear(
            self.llm_model.config.hidden_size, 
            self.llm_model.config.hidden_size
        )
        
        # 冻结视觉模型和文本模型的参数，仅训练图像向文本对齐的那两个全连接层的权重
        # 也就是视觉模型用来对齐的那两个全连接层
        if self.config.freeze_vision_model:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        for param in self.llm_model.parameters():
            param.requires_grad = False
        
    # 3. 在forward中确保输入张量在设备上（而非模型初始化时）
    def forward(self, input_ids, labels, pixel_values, attention_mask=None):
        # 显式将输入移到模型所在设备（而非固定cuda）
        device = self.llm_model.device  # 动态获取模型当前设备
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        pixel_values = pixel_values.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        text_embeds = self.llm_model.get_input_embeddings()(input_ids)
        
        image_embeds = self.vision_model.vision_model(pixel_values).last_hidden_state 
        b, s, d = image_embeds.shape
        image_embeds = image_embeds.view(b, -1, d*4)  # (b, 196, d) --> (b, 49, d*4) 压缩图片tokens
        image_features = self.linear2(F.silu(self.linear1(image_embeds)))
        
        text_embeds = text_embeds.to(image_features.dtype)
        
        inputs_embeds = self.merge_input_ids_with_image_features(image_features, text_embeds, input_ids)
        outputs = self.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs[0]
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)), labels.view(-1).to(logits.device)
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)
        
    def merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids):
        
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_indices, image_indices = torch.where(input_ids == self.tokenizer('<|image_pad|>')['input_ids'][0])
        
        inputs_embeds[batch_indices, image_indices] = image_features.view(-1, embed_dim)
        
        return inputs_embeds
    
class MyDataset(Dataset):
    def __init__(self, images_path, data_path, tokenizer, processor, config):
        super().__init__()
        self.data_path = data_path
        self.images_path = images_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.datas = json.load(f)   
        
            
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        sample = self.datas[index]
        try:
            image_name = sample['image']
            conversations = sample['conversations']
            q_text = self.tokenizer.apply_chat_template([{"role":"system", "content":'You are a helpful assistant.'}, {"role":"user", "content":conversations[0]['value']}], \
                tokenize=False, \
                add_generation_prompt=True).replace('<image>', '<|image_pad|>'*self.config.image_pad_num)
            a_text = conversations[1]['value'] + self.tokenizer.eos_token
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            labels = [tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            
            # 确保序列长度一致
            if len(input_ids) > 0 and len(labels) > 0:
                input_ids = input_ids[:-1]
                labels = labels[1:]
            
            image = Image.open(os.path.join(self.images_path, image_name)).convert("RGB")
            # 处理图像时返回张量格式
            pixel_values = self.processor(text=None, images=image, return_tensors="pt")['pixel_values'][0]
            
        except Exception as e:
            print(f"处理样本 {index} 时出错: {e}")
            # 使用默认图像和文本
            default_image = Image.new('RGB', (224, 224), color='white')
            pixel_values = self.processor(text=None, images=default_image, return_tensors="pt")['pixel_values'][0]
            
            q_text = self.tokenizer.apply_chat_template(
                [{"role":"system", "content":'You are a helpful assistant.'}, 
                 {"role":"user", "content":"图片内容是什么\n<image>"}], 
                tokenize=False, 
                add_generation_prompt=True
            ).replace('<image>', '<|image_pad|>'*self.config.image_pad_num)
            
            a_text = '图片内容为空' + self.tokenizer.eos_token
            q_input_ids = self.tokenizer(q_text, add_special_tokens=False)['input_ids']
            a_input_ids = self.tokenizer(a_text, add_special_tokens=False)['input_ids']
            input_ids = q_input_ids + a_input_ids
            labels = [self.tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            
            if len(input_ids) > 0 and len(labels) > 0:
                input_ids = input_ids[:-1]
                labels = labels[1:]
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values
        } 


class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(feature['input_ids']) for feature in features)
        input_ids = []
        labels = []
        pixel_values = []
        
        for feature in features:
            input_ids.append(feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['input_ids'])))
            labels.append(feature['labels'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['labels'])))
            pixel_values.append(feature['pixel_values'])
            
        # 转换为张量（会在Trainer中自动移到GPU）
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'pixel_values': torch.stack(pixel_values)
        }
            
        
        
if __name__ == '__main__':
    # 打印GPU信息，确认使用RTX 4090
    print(f"使用的GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
    
    config = VLMConfig(vision_model_path=LOCAL_VISION_MODEL_BASE, image_pad_num=49)
    
    model = VLM(config).cuda()
    print(model)
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    
    images_path = './dataset/LLaVA-CC3M-Pretrain-595K/images'
    data_path = './dataset/Chinese-LLaVA-Vision-Instructions/LLaVA-CC3M-Pretrain-595K/chat-translated.json'
    
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path, local_files_only=True)
    processor = AutoProcessor.from_pretrained(config.vision_model_path, local_files_only=True)

    # 确保tokenizer有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 训练参数优化（适合RTX 4090）
    output_dir = 'save/pretrain' 
    args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        per_device_train_batch_size=8,  # RTX 4090显存大，可以调大批次
        learning_rate=1e-4,
        num_train_epochs=5,
        save_steps=500,
        save_total_limit=2,
        fp16=True,  # RTX 4090支持FP16加速
        gradient_accumulation_steps=4,  # 适当减小累积步数，加快训练
        logging_steps=100,
        report_to='tensorboard',
        dataloader_pin_memory=True,  # 启用.pin_memory加速数据传输
        dataloader_num_workers=4,  # 4090可以支持更多数据加载线程
        optim="adamw_torch_fused",  # 使用融合优化器，加速训练
        torch_compile=True,  # 启用TorchCompile加速
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=MyDataset(images_path, data_path, tokenizer, processor, config),
        data_collator=MyDataCollator(tokenizer)  
    )
    
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('save/pretrain')
    trainer.save_state()
    