from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from dataset import SFTDataset
from utils import compute_fkl, compute_rkl, compute_skewed_fkl, compute_skewed_rkl
import os

# 消除 tokenizers 并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class KGTrainer(Trainer):
    def __init__(
        self,
        model = None,
        teacher_model = None,
        if_use_entropy = False,
        args = None,
        data_collator = None, 
        train_dataset = None,
        eval_dataset = None,
        tokenizer = None,
        model_init = None, 
        compute_metrics = None, 
        callbacks = None,
        optimizers = (None, None), 
        preprocess_logits_for_metrics = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.teacher_model = teacher_model
        self.if_use_entropy = if_use_entropy
        
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        
        outputs = model(**inputs)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        
        loss = outputs.loss
        logits = outputs.logits
        teacher_logits = teacher_outputs.logits
        
        # 如果教师模型和学生模型输出形状不匹配，对学生模型进行padding或对教师模型进行截断
        if logits.shape[-1] != teacher_logits.shape[-1]:
            # gap = teacher_logits.shape[-1] - logits.shape[-1]
            # if gap > 0:
            #     pad_logits = torch.zeros((logits.shape[0], logits.shape[1], gap)).to(logits.device)
            #     logits = torch.cat([logits, pad_logits], dim=-1)
            
            teacher_logits = teacher_logits[:, :, :logits.shape[-1]]
        
        labels = inputs['labels']
        kl = compute_fkl(logits, teacher_logits, labels, padding_id=-100, temp=2.0).mean()
        
        if self.if_use_entropy:
            loss_total = 0.5 * kl + 0.5 * loss
        else:
            loss_total = kl
        
        return (loss_total, outputs) if return_outputs else loss_total
        

if __name__ == '__main__':
    
    # 学生模型
    # 显存优化：学生模型也使用 gradient checkpointing，且需要处理梯度
    model = AutoModelForCausalLM.from_pretrained(
        "/root/autodl-tmp/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # 显存优化：开启梯度检查点时必须启用输入梯度
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    lora_config = LoraConfig(
    r=8,  
    lora_alpha=256,  
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1, 
    task_type=TaskType.CAUSAL_LM)
    # 使用lora方法训练
    model = get_peft_model(model, lora_config)
    # model.cuda() # device_map="auto" 已经处理了设备分配
    print(model.print_trainable_parameters())
    
    tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/Qwen2.5-0.5B-Instruct")
    # 消除警告：同步 pad_token_id 并禁用 cache
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    
    # 教师模型，在给定数据上通过lora微调
    # 使用 4-bit 量化加载教师模型以节省显存
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        "/root/autodl-tmp/Qwen2.5-7B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    # 是否加载lora模型
    lora_path = './results_sft_lora/checkpoint-625/' # 请确认此路径是否存在，如果不存在需修改
    # 检查 lora 路径是否存在，不存在则尝试用 saves_sft_lora
    if not os.path.exists(lora_path):
        if os.path.exists('./saves_sft_lora'):
            lora_path = './saves_sft_lora'
            print(f"Checkpoint path not found, using {lora_path} instead.")
        else:
             print(f"WARNING: LoRA path {lora_path} does not exist!")

    teacher_model = PeftModel.from_pretrained(teacher_model, lora_path)
    # teacher_model.cuda() # 4-bit 量化会自动分配 device，无需手动 .cuda()
    teacher_model.eval()
    
    args = TrainingArguments(output_dir='./results', 
                            num_train_epochs=3, 
                            do_train=True, 
                            per_device_train_batch_size=1, # 显存优化：减小 batch size
                            gradient_accumulation_steps=16, # 显存优化：增加累积步数
                            logging_steps=10,
                            report_to='tensorboard',
                            save_strategy='epoch',
                            save_total_limit=3,
                            bf16=True,
                            learning_rate=3e-4, # 降低学习率
                            lr_scheduler_type='cosine',
                            warmup_ratio=0.1,
                            max_grad_norm=0.3,
                            dataloader_num_workers=2, # 显存优化：减少 worker 数量
                            dataloader_pin_memory=True,
                            gradient_checkpointing=True, # 显存优化：开启梯度检查点
                            gradient_checkpointing_kwargs={'use_reentrant': False},
                            optim="paged_adamw_32bit" # 显存优化：使用 Paged AdamW
                            )
    data_collator = DefaultDataCollator()
    dataset = SFTDataset('./data_zh.json', tokenizer=tokenizer, max_seq_len=512)
    trainer = KGTrainer(model=model,
                        teacher_model=teacher_model, 
                        if_use_entropy = True,
                        args=args, 
                        train_dataset=dataset, 
                        tokenizer=tokenizer, 
                        data_collator=data_collator)
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./saves')
    trainer.save_state()
    
      
    