from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from copy import deepcopy
from datasets import load_dataset
from reward_func import *
from torch.amp import autocast
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class GSM8KDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        
        self.tokenizer = tokenizer
        data = load_dataset(data_path)
        self.data = data['train']
  
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        # prompt = self.tokenizer.apply_chat_template(sample['prompt'], tokenize=False, add_generation_prompt=True)
        answer = sample['answer_only']
        prompt = sample['question_zh-cn']
        return {'prompt': prompt, 'answer': answer}


@dataclass
class Samples:
    prompt_response_ids: torch.Tensor
    response_ids: torch.Tensor
    prompt: Any
    answer: Any
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    response_length: int


class GRPOArguments:
    
    output_dir = './output'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 0.000001
    save_steps = 100
    epoch = 3
    num_generations = 1 # 组内样本数，减少到1以节省显存
    max_prompt_length = 256 # 最大输入长度
    max_generate_length = 256 # 最大输出长度
    reward_weights : List[float] = None # 奖励的权重（多个奖励函数）
    beta = 0.0 # KL 散度的系数，为 0 则忽略 KL 散度，即不使用参考模型
    clip_eps = 0.2
    gradient_accumulation_steps = 4 # 梯度累加
    num_iterations = 1 # 采样一次样本训练模型轮数
    batch_size = 1
    use_bf16 = True # 使用 BF16 混合精度训练（兼容性更好）

class GRPOTrainer:
    def __init__(self,
        model = None,
        reward_funcs: Union[List[str], List[Callable]] = None,
        args = None,
        train_dataset: Optional[Union[Dataset]] = None,
        eval_dataset: Optional[Union[Dataset]] = None,
        tokenizer = None,
        reward_tokenizers = None):

        self.args = args
        # 加载模型
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float32)
        else:
            # 使用 BF16 精度加载模型（BF16 动态范围比 FP16 更大，兼容性更好）
            if self.args.use_bf16 and torch.cuda.is_available():
                model = model.to(dtype=torch.bfloat16)
        
        self.model = model.to(self.args.device)
        
        # 启用梯度检查点以节省显存
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        # 禁用 use_cache（与梯度检查点不兼容）
        if hasattr(self.model, 'config'):
            self.model.config.use_cache = False
        
        # 设置模型为训练模式
        self.model.train()
        
        # 是否使用参考模型
        self.ref_model = None
        if self.args.beta != 0.0:
            self.ref_model = deepcopy(model)
            self.ref_model.eval()
    
        
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        
        self.tokenizer = self.get_tokenizer(tokenizer)
        
        
        if isinstance(reward_funcs, str):
            reward_funcs = [reward_funcs]
        
        for i, reward_func in enumerate(reward_funcs):
            # 如果奖励函数为字符串，表示使用的是奖励模型，则加载模型
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1).to(self.args.device)
        
        self.reward_funcs = reward_funcs
        
        if reward_tokenizers is None:
            reward_tokenizers = [None] * len(reward_funcs)
            
        elif isinstance(reward_tokenizers, str):
            reward_tokenizers = [reward_tokenizers]
            
        else:
            if len(reward_tokenizers) != len(reward_funcs):
                raise ValueError("Length of reward_tokenizers must be equal to the number of reward_funcs.")
            
        for i, (reward_tokenizer, reward_func) in enumerate(zip(reward_tokenizers, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_tokenizer is None:
                    reward_tokenizer = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_tokenizer.pad_token_id is None:
                    reward_tokenizer.pad_token = reward_tokenizer.eos_token
                
                reward_func.config.pad_token_id = reward_tokenizer.pad_token_id
                reward_tokenizers[i] = reward_tokenizer
        self.reward_tokenizers = reward_tokenizers
        
        # 清理缓存后再创建优化器
        torch.cuda.empty_cache()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # 缓存已经生成的数据的一个批次的数据，可供模型多次训练迭代，无需重新生成
        self.input_buffer = [None] * self.args.gradient_accumulation_steps
        
        # 模型更新的次数
        self.update_steps = 0 
    def get_tokenizer(self, tokenizer):
        tokenizer.padding_side = "left"
        return tokenizer
    
    # 生成样本，以组为单位
    def generate_samples(self, inputs):
        samples_list = []
        self.model.eval()
        prompts = [prompt for prompt in inputs['prompt']]
        answers = [None] * len(prompts)
        
        if 'answer' in inputs:
            answers = [answer for answer in inputs['answer']]
        
        max_length = self.args.max_generate_length + self.args.max_prompt_length
        
        # 确保 tokenizer 正确设置了 pad_token_id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        for prompt, answer in zip(prompts, answers):
            # 应用聊天模板，加入系统提示词
            input_text = self.tokenizer.apply_chat_template([{"role": "system", 'content': SYSTEM_PROMPT}, {"role": "user", 'content': prompt}], add_generation_prompt=True, tokenize=False)
            
            # 生成一个 group 的输入数据
            inputs_tokenized = self.tokenizer(
                [input_text] * self.args.num_generations, 
                padding='max_length', 
                max_length=self.args.max_prompt_length, 
                truncation=True, 
                return_tensors='pt', 
                add_special_tokens=True
            )
            
            # 确保 input_ids 是正确的 dtype (int64)
            prompt_ids = inputs_tokenized['input_ids'].to(dtype=torch.long, device=self.args.device)
            attention_mask = inputs_tokenized['attention_mask'].to(dtype=torch.long, device=self.args.device)
            
            with torch.no_grad():
                # 使用带温度的采样生成
                prompt_response_ids = self.model.generate(
                    input_ids=prompt_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.args.max_generate_length,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=False
                ).sequences
                
            # 截断或填充到固定长度
            if prompt_response_ids.size(1) >= max_length:
                prompt_response_ids = prompt_response_ids[:, :max_length]
            else:
                padding_length = max_length - prompt_response_ids.size(1)
                padding = torch.full(
                    (prompt_response_ids.size(0), padding_length),
                    fill_value=self.tokenizer.pad_token_id,
                    dtype=prompt_response_ids.dtype,
                    device=prompt_response_ids.device
                )
                prompt_response_ids = torch.cat([prompt_response_ids, padding], dim=1)
          
            attention_mask = (prompt_response_ids.ne(self.tokenizer.pad_token_id)).to(dtype=torch.long)
            response_ids = prompt_response_ids[:, prompt_ids.size(1):]
            action_mask = (response_ids.ne(self.tokenizer.eos_token_id) & response_ids.ne(self.tokenizer.pad_token_id)).to(dtype=torch.long)
        

            # 存储的是一个 group 的数据
            samples = Samples(
                prompt_response_ids=prompt_response_ids,
                response_ids=response_ids,
                prompt = prompt,
                answer = answer,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                response_length=action_mask.float().sum(dim=-1)
            )
            samples_list.append(samples)
            
            # 每生成一个样本后清理缓存
            torch.cuda.empty_cache()

        return samples_list
    
    # 生成经验(优势、token的概率分布)
    def generate_experiences(self, inputs):
        
        self.model.eval()
        samples_list = self.generate_samples(inputs)
        
        batch_prompt_response_ids = []
        batch_attention_mask = []
        batch_action_mask = []
        batch_advantages = []
        batch_old_action_log_probs = []
        batch_ref_action_log_probs = []
        
        for samples in samples_list:
            prompt_response_ids = samples.prompt_response_ids # shape: (num_generations, seq_len)
            response_ids = samples.response_ids # shape: (num_generations, seq_len)
            answer = samples.answer
            attention_mask = samples.attention_mask # shape: (num_generations, seq_len)
            action_mask = samples.action_mask # shape: (num_generations, seq_len)
            num_actions = samples.num_actions
            prompt = samples.prompt
            batch_prompt_response_ids.append(prompt_response_ids)
            batch_attention_mask.append(attention_mask)
            batch_action_mask.append(action_mask)
            
            with torch.no_grad():
                # 计算策略模型输出 token 的概率
                old_action_log_probs = self.get_action_log_probs(self.model, prompt_response_ids, attention_mask, num_actions)
                batch_old_action_log_probs.append(old_action_log_probs)
                
                # 是否使用参考模型
                if self.ref_model:
                    #计算参考模型输出 token 的概率
                    ref_action_log_probs = self.get_action_log_probs(self.ref_model, prompt_response_ids, attention_mask, num_actions)
                    batch_ref_action_log_probs.append(ref_action_log_probs)
                
                # 存储各个奖励函数在一个 group 内各个响应的奖励
                rewards_per_func = torch.zeros(len(self.reward_funcs), self.args.num_generations, device=self.args.device)
                
                # 将输出转换成文本
                response_texts = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                prompt_texts = [prompt] * len(response_texts)
                prompt_response_texts = [prompt + response for prompt, response in zip(prompt_texts, response_texts)]
                
                for i, (reward_func, reward_tokenizer) in enumerate(
                    zip(self.reward_funcs, self.reward_tokenizers)
                ):
                    if isinstance(reward_func, PreTrainedModel):
                        with torch.inference_mode():
                            reward_model_inputs = reward_tokenizer(prompt_response_texts, return_tensors="pt", padding=True)
                            rewards_per_func[i] = reward_func(**reward_model_inputs.to(self.args.device)).logits.squeeze(-1)
                    
                    else:
                        answers = [answer] * len(prompt_texts)
                        output_reward_func = reward_func(prompts=prompt_texts, responses=response_texts, answers=answers)
                        output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                        rewards_per_func[i] = torch.tensor(output_reward_func, dtype=torch.float32, device=self.args.device)
                
                # rewards_per_func: [num_funcs, num_generations]
                if not self.args.reward_weights:
                    self.args.reward_weights = [1.0] * len(self.reward_funcs)
                if len(self.args.reward_weights) != len(self.reward_funcs):
                    raise ValueError("The number of reward weights must be equal to the number of reward functions.")
                # 乘以各个奖励函数的权重
                rewards = rewards_per_func * torch.tensor(self.args.reward_weights, dtype=torch.float32, device=rewards_per_func.device).unsqueeze(1)
                
                # rewards: [num_funcs, num_generations]
                rewards = rewards.sum(dim=0) # shape: [num_generations]
                print(f'rewards: {rewards}')
                mean_group_rewards = rewards.mean()
                std_group_rewards = rewards.std()
                
                # GRPO 的优势是句子粒度的，而非 token 粒度的
                advantages = (rewards - mean_group_rewards) / (std_group_rewards + 1e-8) # shape: [num_generations]
                batch_advantages.append(advantages)
            
            # 处理完一个样本后清理缓存
            torch.cuda.empty_cache()
        
               
        return {
            "prompt_response_ids": torch.cat(batch_prompt_response_ids, dim=0),
            "attention_mask": torch.cat(batch_attention_mask, dim=0),
            "action_mask": torch.cat(batch_action_mask, dim=0),
            "old_action_log_probs": torch.cat(batch_old_action_log_probs, dim=0),
            "ref_action_log_probs": torch.cat(batch_ref_action_log_probs, dim=0) if self.ref_model else None,
            "advantages": torch.cat(batch_advantages, dim=0),
        }
    
    def compute_loss(self, model, inputs):
        
        prompt_response_ids = inputs['prompt_response_ids']
        attention_mask = inputs['attention_mask']
        action_mask = inputs['action_mask']
        num_actions = action_mask.size(1)
        
        # 过滤掉 action_mask 全为 0 的样本
        valid_mask = action_mask.sum(dim=1) > 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=self.args.device, requires_grad=True)
        
        prompt_response_ids = prompt_response_ids[valid_mask]
        attention_mask = attention_mask[valid_mask]
        action_mask = action_mask[valid_mask]
        advantages = inputs['advantages'][valid_mask]
        
        action_log_probs = self.get_action_log_probs(model, prompt_response_ids, attention_mask, num_actions)
        
        if self.args.beta != 0.0:
            
            ref_action_log_probs = inputs['ref_action_log_probs']
            log_ratio = ref_action_log_probs - action_log_probs 
            log_ratio = log_ratio * action_mask
            
            # k3: log_ratio.exp() - 1 - log_ratio
            k3 = log_ratio.exp() - 1 - log_ratio
        
        old_action_log_probs = inputs['old_action_log_probs'] if self.args.num_iterations > 1 else action_log_probs.detach()
        
        # 过滤 old_action_log_probs
        old_action_log_probs = old_action_log_probs[valid_mask]
        
        # 防止数值不稳定：限制 log_probs 差异
        log_prob_diff = (action_log_probs - old_action_log_probs).clamp(-10, 10)
        coef_1 = torch.exp(log_prob_diff)
        coef_2 = torch.clamp(coef_1, 1 - self.args.clip_eps, 1 + self.args.clip_eps)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        per_token_loss = per_token_loss * action_mask
        if self.args.beta != 0.0:
            per_token_loss = per_token_loss + self.args.beta * k3
        
        # 避免除以零
        denom = action_mask.sum(dim=1)
        denom = denom.clamp(min=1)
        loss = per_token_loss.sum(dim=1) / denom
        loss = loss.mean()
        
        # 检查 loss 是否有效
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=self.args.device, requires_grad=True)
        
        return loss


    def get_action_log_probs(self, model, input_ids, attention_mask, num_actions):
        
        # 使用 FP32 计算以保证数值稳定性（log_probs 需要高精度）
        # 模型权重是 FP16，但计算时会自动转换
        output = model(input_ids, attention_mask=attention_mask)
        logits = output.logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        log_probs_labels = log_probs.gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1))
        action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]
        return action_log_probs

    
    
    def train_step(self, model, inputs, optimizer, step):
        model.train()
        
        # 使用 BF16 混合精度训练（不需要 GradScaler）
        if self.args.use_bf16:
            with autocast('cuda', dtype=torch.bfloat16):
                loss = self.compute_loss(model, inputs)
            loss = loss / self.args.gradient_accumulation_steps
            loss.backward()
        else:
            loss = self.compute_loss(model, inputs)
            loss = loss / self.args.gradient_accumulation_steps
            loss.backward()
            
        if (step + 1) % self.args.gradient_accumulation_steps == 0:
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
            writer.add_scalar("grpo_loss", loss.item(), self.update_steps)
            print(f"step: {self.update_steps}/{self.global_steps}  grpo_loss: {loss.item():.8f}")
        
        # 清理缓存
        del loss
        torch.cuda.empty_cache()

    def train(self):
        self.global_steps = self.args.num_iterations * self.args.epoch * len(self.train_dataset) // (self.args.batch_size * self.args.gradient_accumulation_steps)
        
        # 初始化时清理缓存
        torch.cuda.empty_cache()
        
        for epoch in range(self.args.epoch):
            print(f"Epoch {epoch + 1}/{self.args.epoch}")
            
            dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
            for idx, batch in enumerate(dataloader):
                
                inputs = self.generate_experiences(batch)
                self.input_buffer[idx % self.args.gradient_accumulation_steps] = inputs
                if (idx + 1) % self.args.gradient_accumulation_steps == 0:
                   
                    for _ in range(self.args.num_iterations):
                        for step, inputs in enumerate(self.input_buffer):
                            self.train_step(self.model, inputs, self.optimizer, step)
                        
                        self.update_steps += 1
                        if self.update_steps % self.args.save_steps == 0:
                            self.model.save_pretrained(self.args.output_dir + f'/checkpoint_{self.update_steps}')
                            self.tokenizer.save_pretrained(self.args.output_dir + f'/checkpoint_{self.update_steps}')
                
                # 定期清理缓存
                if idx % 5 == 0:
                    torch.cuda.empty_cache()
                
                del inputs
            
            # 每个 epoch 后清理缓存
            torch.cuda.empty_cache()
    def save_model(self):
        self.model.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)           

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    SYSTEM_PROMPT = """
                    按照如下格式回答问题：
                    <think>
                    你的思考过程
                    </think>
                    <answer>
                    你的回答
                    </answer>
                    """
    
    args = GRPOArguments()
    
    writer = SummaryWriter('./runs')
    # 策略模型
    tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/Qwen2.5-1.5B-Instruct')
    
    # 使用 BF16 加载模型（兼容性更好，不需要 GradScaler）
    model = AutoModelForCausalLM.from_pretrained(
        '/root/autodl-tmp/Qwen2.5-1.5B-Instruct',
        torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float32
    )
    
    # 初始化时清理缓存
    torch.cuda.empty_cache()
    # 奖励函数
    # reward_model = '/root/autodl-tmp/reward-model-deberta-v3-large-v2'
    # reward_tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/reward-model-deberta-v3-large-v2')
    
    prompts_dataset = GSM8KDataset('/root/autodl-tmp/deepseek_learn/deepseek_r1_train/gsm8k_chinese', tokenizer)
  
    trainer = GRPOTrainer(model=model,
                          reward_funcs = [correctness_reward, digit_reward, hard_format_reward, mark_reward],
                          args=args,
                          train_dataset=prompts_dataset,
                          tokenizer=tokenizer)
    trainer.train()
    trainer.save_model()
    