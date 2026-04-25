import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import os
import pandas as pd

from torch.utils.data import IterableDataset, Dataset
import json
import numpy as np
from transformers import  PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PretrainedConfig
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator, DataCollatorForTokenClassification, AutoConfig
from dataset import SFTDataset, LLMDataset


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.float()
    
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotate_pos_emb(q, k, cos, sin, unsqueeze_dim=2):
    
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    k_embed = (k*cos) + (rotate_half(k)*sin)

    # 训练或prefill
    if q.shape[1] == k.shape[1]:
        q_embed = (q*cos) + (rotate_half(q)*sin)
    
    # 推理generate（逐token）
    else:
 
        q_embed = (q*cos[:,-1:]) + (rotate_half(q)*sin[:,-1:])
   
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=1024):
        super(RotaryEmbedding, self).__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float().unsqueeze(1)
        freqs = t @ inv_freq.unsqueeze(0)
        freqs = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())
        
    def forward(self, q, k):
        cos = self.cos_cached[:q.shape[1], :].unsqueeze(0)
        sin = self.sin_cached[:q.shape[1], :].unsqueeze(0)
        return apply_rotate_pos_emb(q, k, cos, sin)
    
def repeat_kv(hidden_states, n_rep):
    
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


class DynamicCache:
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.key_cache = [None for _ in range(config.n_layers)]
        self.value_cache = [None for _ in range(config.n_layers)]

class Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.k_cache, self.v_cache = None, None
        self.is_causal = True
        self.flash_attn = self.config.flash_attn
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.residual_dropout = nn.Dropout(self.dropout)
        self.attention_dropout = nn.Dropout(self.dropout)
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        
    def forward(self, hidden_states, 
                attention_mask: torch.Tensor=None, 
                cache_params=None):
        
        b, s = hidden_states.shape[:2]

        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        if cache_params is not None:
            if cache_params.key_cache[self.layer_idx] is None:
                cache_params.key_cache[self.layer_idx] = k
                cache_params.value_cache[self.layer_idx] = v
            
            else:
                k = torch.cat([cache_params.key_cache[self.layer_idx], k], dim=1)
                v = torch.cat([cache_params.value_cache[self.layer_idx], v], dim=1)
                cache_params.key_cache[self.layer_idx] = k
                cache_params.value_cache[self.layer_idx] = v
                
        q = q.view(b, q.shape[1], self.num_heads, self.head_dim)
        k = k.view(b, k.shape[1], self.num_key_value_heads, self.head_dim)
        v = v.view(b, v.shape[1], self.num_key_value_heads, self.head_dim)
        
        q, k = self.rotary_emb(q, k)
        
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        
        q = q.transpose(1, 2) # b, self.num_heads, s, self.head_dim
        k = k.transpose(1, 2) # b, self.num_heads, s, self.head_dim
        v = v.transpose(1, 2) # b, self.num_heads, s, self.head_dim
        
        if self.flash_attn and s > 1:
            attn_mask = None
            if attention_mask is not None:
                mask = torch.triu(torch.ones(b, s, s), diagonal=1).bool().to(attention_mask.device) # [b,s,s]
                attention_mask = attention_mask.bool().unsqueeze(-1) # [b,s,1]
                attn_mask = mask | ~attention_mask 
                
                attn_mask = attn_mask.float().masked_fill(attn_mask, float('-inf')).unsqueeze(1)
              
                
            
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, 
                                                    dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal) 
            
            
        else:
            mask = torch.triu(torch.ones(b, s, s), diagonal=1).bool()
            
            if attention_mask is not None:
                mask = mask.to(attention_mask.device)
                mask = mask | ~attention_mask.bool().unsqueeze(-1)
            
            mask = mask.float().masked_fill(mask, float('-inf')).unsqueeze(1)
           
            scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)  # 计算注意力分数
          
            scores = scores + mask  # 应用掩码
            scores = F.softmax(scores.float(), dim=-1).type_as(q)  # 计算 softmax
            scores = self.attention_dropout(scores)  # 应用注意力 dropout
            output = torch.matmul(scores, v)  # 计算输出
        
        output = output.transpose(1, 2).contiguous().view(b, s, -1) # b, s, self.hidden_size
        
        output = self.o_proj(output)
        output = self.residual_dropout(output)
        return output
    
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        
    def forward(self, x):
        
        down_proj = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

def load_balancing_loss_func(
    gate_logits,
    num_experts,
    top_k):
    concatenated_gate_logits = torch.cat([layer_gate for layer_gate in gate_logits], dim=0) # 各个层的gate_logit进行合并[layers X batch_size X sequence_length, num_experts]
    routing_weights = F.softmax(concatenated_gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
    
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)


    router_prob_per_expert = torch.mean(routing_weights, dim=0)
    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts
    
class Gating(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.topk = config.topk
        self.expert_num = config.expert_num
        self.gate = nn.Linear(self.hidden_size, self.expert_num)
        
    def forward(self, x):
        # x dim: b, s, hidden_size
        logits = self.gate(x)  # gate: b, s, expert_num
        logits_topk, indices = logits.topk(self.topk, dim=-1) # 选择概率最大的两个专家，返回两个专家对每个token的概率
        zeros = torch.full_like(logits, float("-inf")) # 创建一个全为负无穷的矩阵，用于屏蔽其他专家的概率并重新归一化概率最大的两个专家
        sparse_logits = zeros.scatter(dim=-1, index=indices, src=logits_topk)  # 将选择的两个专家的概率按指定索引填充
        sparse_logits = F.softmax(sparse_logits, dim=-1) # 得到一个稀疏矩阵，选择的两个专家对每个token的概率和为1
        gate_logit = logits.view(-1, self.expert_num)
        
        return sparse_logits, indices, gate_logit
    
class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias) 
    def forward(self, x):
        down_proj = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.expert_num)])
        self.gating = Gating(config)
        
    def forward(self, x):
        sparse_logits, indices, gate_logit = self.gating(x)
        final_outputs = torch.zeros_like(x) 
        x_flat = x.view(-1, x.shape[-1])  # (batch_size * seq_len, dim)
        sparse_logits_flat = sparse_logits.view(-1, sparse_logits.shape[-1])  # (batch_size * seq_len, export_num))
        
        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(-1)  # (batch_size, seq_len)
            expert_mask_flat = expert_mask.view(-1) # (batch_size * seq_len)
            if expert_mask_flat.any():
                expert_input = x_flat[expert_mask_flat]  # (seq_true, dim)
                export_output = expert(expert_input)  # (seq_true, dim)
                
                gate_scores = sparse_logits_flat[expert_mask_flat, i].unsqueeze(1)  # (seq_true) --> (seq_true, 1)
                
                weighted_output = export_output * gate_scores  # (seq_true, dim)
                
                final_outputs[expert_mask] += weighted_output
                
        
        return final_outputs, gate_logit
        
        
# 修改残差连接部分
class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config, layer_idx)
        self.moe = MoE(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
        self.layer_idx = layer_idx

        # 为每一层添加独立的query
        self.attn_query = nn.Parameter(torch.randn(1, 1, self.hidden_size) * 0.02)
        self.mlp_query  = nn.Parameter(torch.randn(1, 1, self.hidden_size) * 0.02)

        # 对历史层做归一化
        self.attn_norm = RMSNorm(self.hidden_size)
        self.mlp_norm = RMSNorm(self.hidden_size)

    def attn_res(
        self,
        past_layer_states: List[torch.Tensor],   # 前面所有层的输出
        current_state: torch.Tensor,
        query: torch.Tensor,   # shape: [1,1,d]
        norm: RMSNorm

    ) -> torch.Tensor:
        
        # values保留原始输出
        values = torch.stack(past_layer_states + [current_state], dim=2) # shape: [b, s, layer, d]
        # keys做标准化
        keys = norm(values)

        if query.dim() == 3:
            query = query.unsqueeze(2)  # shape: [1, 1, 1, d]

        scores = (keys * query).sum(dim=-1) # shape: [b, s, layer]
        weights = F.softmax(scores, dim=-1) # shape: [b, s, layer]

        weighted_values = weights.unsqueeze(-1) * values  # shape: [b, s, layer, 1] * [b, s, layer, d] = [b, s, layer, d]
        aggregated = weighted_values.sum(dim=2)           # shape: [b, s, d]

        return aggregated

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_layer_states: List[torch.Tensor],
        attention_mask: torch.Tensor = None,
        cache_params:DynamicCache = None
    ):
        

        attn_input = self.attn_res(past_layer_states, hidden_states, self.attn_query, self.attn_norm)
        past_layer_states.append(hidden_states)
        
        attn_normed = self.input_layernorm(attn_input)

        hidden_states = self.self_attn(hidden_states=attn_normed, attention_mask=attention_mask, cache_params=cache_params)

        mlp_input = self.attn_res(past_layer_states, hidden_states, self.mlp_query, self.mlp_norm)
        past_layer_states.append(hidden_states)
        mlp_normed = self.post_attention_layernorm(mlp_input)
        if self.layer_idx % 2 == 0:
            hidden_states = self.mlp(mlp_normed)
            gate_logit = None
        else:
            hidden_states, gate_logit = self.moe(mlp_normed)
        
        return hidden_states, gate_logit, past_layer_states
   
class Config(PretrainedConfig):
    model_type = "moe_model"
    
    def __init__(self,
                hidden_size = 512,
                num_attention_heads = 16,
                num_key_value_heads = 8,
                flash_attn = True,
                attention_bias = False,
                max_seq_len = 512,
                intermediate_size = 2048,
                mlp_bias = False,
                vocab_size = 6400,
                n_layers = 8,
                dropout = 0.0,
                expert_num = 4,
                topk = 2,
                output_router_logits = True,
                aux_loss_coef = 0.01,
                **kwargs):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.flash_attn = flash_attn
        self.attention_bias = attention_bias
        self.max_seq_len = max_seq_len
        self.intermediate_size = intermediate_size
        self.mlp_bias = mlp_bias
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.expert_num = expert_num
        self.topk = topk
        self.output_router_logits = output_router_logits
        self.aux_loss_coef = aux_loss_coef
        super().__init__(**kwargs)


class LLM(PreTrainedModel):
    config_class = Config
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vocab_size = self.config.vocab_size
        self.n_layers = self.config.n_layers
        self.expert_num = self.config.expert_num
        self.topk = self.config.topk
        
        self.tokon_embeddings = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.dropout) 
        self.layers = torch.nn.ModuleList() 
        for layer_idx in range(self.n_layers):
            self.layers.append(DecoderLayer(self.config, layer_idx)) 
        self.norm = RMSNorm(self.config.hidden_size)
        self.output = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False) 
        self.tokon_embeddings.weight = self.output.weight
        self.apply(self._init_weights) 
        self.loss = None 
        self.aux_loss = None
        
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers)) 

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) 
            
        
    def forward(self, input_ids, attention_mask=None, labels=None, cache_params = None):
        
        all_router_logits = () if self.config.output_router_logits else None
        hidden_states = self.tokon_embeddings(input_ids) 
 
        hidden_states = self.dropout(hidden_states)  

        # 历史层的状态
        past_layer_states = []


        for idx, layer in enumerate(self.layers):
            hidden_states, gate_logit, past_layer_states= layer(hidden_states, past_layer_states, attention_mask = attention_mask, cache_params=cache_params)
            if gate_logit is not None:
                all_router_logits += (gate_logit, )  
            
        hidden_states = self.norm(hidden_states) 
        
        
        if labels is not None:
            logits = self.output(hidden_states)  
            self.loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=0) 
        else:
            logits = self.output(hidden_states[:, [-1], :])  
            self.loss = None  
        
        if self.config.output_router_logits:
            self.aux_loss = load_balancing_loss_func(all_router_logits, self.expert_num, self.topk)
            
            if labels is not None:
                self.loss += self.config.aux_loss_coef * self.aux_loss.to(self.loss.device)

        return CausalLMOutputWithPast(self.loss, logits)
    
    @torch.inference_mode
    def generate(self, inputs, eos, max_new_tokens, temperature=0.7, top_k=None, stream=True, repetition_penalty=1., use_cache=True):
        
        cache_params = None
        if use_cache:
            cache_params = DynamicCache(self.config)
            
        input_ids = inputs['input_ids']
        attention_mask = None
        labels = None
        s = input_ids.shape[1]
        
        idx_next = None
  
        while input_ids.shape[1] < max_new_tokens - 1:  
            if idx_next is None:
                inference_res = self(input_ids, attention_mask=attention_mask, labels=labels, cache_params=cache_params)  
            else:
                inference_res = self(idx_next, attention_mask=attention_mask, labels=labels, cache_params=cache_params)
            
            logits = inference_res.logits 
            logits = logits[:, -1, :] 

            for token in set(input_ids.tolist()[0]):  
                logits[:, token] /= repetition_penalty

            if temperature == 0.0: 
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature  
                if top_k is not None:  
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf') 

                probs = F.softmax(logits, dim=-1)  
                idx_next = torch.multinomial(probs, num_samples=1, generator=None)  
    

            if idx_next == eos:  
                break

            input_ids = torch.cat((input_ids, idx_next), dim=1)  
            if stream:  
                yield input_ids[:, s:]  

        if not stream:  
            yield input_ids[:, s:]  
               
if __name__ == '__main__':   

    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    config = Config()
    model = LLM(config)
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    data_collator = DefaultDataCollator()
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer", use_fast=True)
    args = TrainingArguments(output_dir='./kimi', 
                            num_train_epochs=1, 
                            do_train=True, 
                            per_device_train_batch_size=32,
                            gradient_accumulation_steps=8,
                            # max_steps=15000,
                            logging_steps=1,
                            save_steps=3000,
                            report_to='tensorboard',
                            save_total_limit=5,
                            bf16=True,
                            learning_rate=2e-4,
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=8,
                            dataloader_pin_memory=True,
                            save_safetensors=False)          
    dataset = LLMDataset('pretrain.jsonl', tokenizer=tokenizer, max_seq_len=512)
    trainer = Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer, data_collator=data_collator)
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./saves/kimi')
    trainer.save_state()