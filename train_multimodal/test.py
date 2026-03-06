from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoConfig
from PIL import Image
import torch
from torch.nn import functional as F
from train import VLMConfig, VLM

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(
    '/root/autodl-tmp/llm_related/train_multimodal_from_scratch/siglip-base-patch16-224'
)
tokenizer = AutoTokenizer.from_pretrained(
    '/root/autodl-tmp/llm_related/train_multimodal_from_scratch/Qwen2.5-0.5B-Instruct'
)

# 确保tokenizer有pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 注册模型配置和类
AutoConfig.register("vlm_model", VLMConfig)
AutoModelForCausalLM.register(VLMConfig, VLM)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    '/root/autodl-tmp/llm_related/train_multimodal_from_scratch/save/sft'
)
model.to(device)
model.eval()

# 构建输入文本
q_text = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": 'You are a helpful assistant.'},
        {"role": "user", "content": '图片中有鸭子吗\n<image>'}
    ],
    tokenize=False,
    add_generation_prompt=True
).replace('<image>', '<|image_pad|>' * 49)

# 处理输入IDs
input_ids = tokenizer(q_text, return_tensors='pt')['input_ids'].to(device)

# 关键修改：构造一个与input_ids形状一致的labels张量（用pad_token_id填充）
# 避免train.py中labels.to(device)报错
labels = torch.full_like(input_ids, fill_value=tokenizer.pad_token_id, dtype=torch.long).to(device)

# 处理图像
image = Image.open(
    '/root/autodl-tmp/llm_related/train_multimodal_from_scratch/test_images/test1.jpg'
).convert("RGB")
pixel_values = processor(text=None, images=image, return_tensors="pt")['pixel_values'].to(device)

# 生成参数
max_new_tokens = 100
temperature = 0.0
eos = tokenizer.eos_token_id
top_k = None
s = input_ids.shape[1]

with torch.no_grad():  # 禁用梯度计算
    while input_ids.shape[1] < s + max_new_tokens - 1:
        # 传入构造的labels（而非None），避免AttributeError
        inference_res = model(input_ids=input_ids, labels=labels, pixel_values=pixel_values)
        logits = inference_res.logits[:, -1, :]

        # 温度调整和采样
        if temperature == 0.0:
            _, idx_next = torch.topk(logits, k=1, dim=-1)
        else:
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        if idx_next == eos:
            break

        # 更新input_ids和labels（保持形状一致）
        input_ids = torch.cat((input_ids, idx_next), dim=1)
        # 新添加的token位置的labels仍用pad_token_id填充（不影响推理）
        labels = torch.cat(
            (labels, torch.tensor([[tokenizer.pad_token_id]], device=device, dtype=torch.long)),
            dim=1
        )

# 解码并输出结果
generated_text = tokenizer.decode(input_ids[:, s:][0], skip_special_tokens=True)
print(generated_text)