import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DefaultDataCollator
from peft import LoraConfig, get_peft_model, TaskType
from dataset import SFTDataset
import os

# 消除 tokenizers 并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train():
    # Model path (replace with your local path or Hugging Face model ID)
    model_path = "./Qwen2.5-7B-Instruct" 
    # Try to use local path if it looks like the user's environment structure, otherwise default
    if os.path.exists("/root/autodl-tmp/Qwen2.5-7B-Instruct"):
        model_path = "/root/autodl-tmp/Qwen2.5-7B-Instruct"

    print(f"Loading model from: {model_path}")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16, # Use bfloat16 for training stability if supported
        device_map="auto",
        trust_remote_code=True
    )

    # 消除警告：同步 pad_token_id 并禁用 cache
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    # 显存优化：开启梯度检查点时必须启用输入梯度
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # LoRA Configuration
    lora_config = LoraConfig(
        r=16, # Rank
        lora_alpha=32, # Alpha
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    data_path = 'data_zh.json'
    dataset = SFTDataset(data_path, tokenizer=tokenizer, max_seq_len=512)

    # 调试：检查第一条数据的 Labels 是否有效
    if len(dataset) > 0:
        sample = dataset[0]
        valid_labels = sample['labels'][sample['labels'] != -100]
        print(f"DEBUG: Sample 0 valid labels count: {len(valid_labels)}")
        if len(valid_labels) > 0:
            print(f"DEBUG: Sample 0 target text: {tokenizer.decode(valid_labels)}")
        else:
            print("WARNING: Sample 0 has NO valid labels (all -100). Check dataset/tokenizer!")

    # 计算 steps_per_epoch
    # batch_size * gradient_accumulation_steps
    steps_per_epoch = len(dataset) // (1 * 16) # per_device_train_batch_size * gradient_accumulation_steps
    save_steps = int(steps_per_epoch * 0.1)
    if save_steps < 1:
        save_steps = 1
    print(f"Total dataset size: {len(dataset)}, Steps per epoch: {steps_per_epoch}, Save steps (0.1 epoch): {save_steps}")

    # Training Arguments
    training_args = TrainingArguments(
        output_dir='./results_sft_lora',
        num_train_epochs=3, # 之前被改成了 1，恢复为 3 或者保持用户想要的，这里假设是 3 因为用户说 0.26 epoch
        per_device_train_batch_size=1, # 显存优化：减小 batch size
        gradient_accumulation_steps=16, # 显存优化：增加累积步数以保持有效 batch size
        learning_rate=3e-4, # 降低学习率防止 Loss 激增
        logging_steps=10,
        save_strategy='steps',
        save_steps=save_steps, # 每 0.1 epoch 保存一次
        save_total_limit=10, # 增加保存数量限制，防止存太多
        bf16=True, # Enable BF16 if supported
        lr_scheduler_type='cosine',
        warmup_ratio=0.1, # 增加 Warmup 比例，让模型更平稳地进入训练状态
        max_grad_norm=0.3, # 梯度裁剪，防止梯度爆炸，有助于 Loss 稳定下降
        report_to='tensorboard',
        dataloader_num_workers=2, # 显存优化：减少 worker 数量
        remove_unused_columns=False, # Important for custom dataset
        gradient_checkpointing=True, # 显存优化：开启梯度检查点
        gradient_checkpointing_kwargs={'use_reentrant': False}, # 避免警告
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=DefaultDataCollator()
    )

    # Start Training
    print("Starting training...")
    trainer.train()

    # Save Model
    save_path = './saves_sft_lora'
    print(f"Saving model to {save_path}")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    train()
