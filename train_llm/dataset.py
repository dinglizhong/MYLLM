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


class LLMDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        
        line = self.data[index]
        line = json.loads(line)
        text = '<s>' + line['text'] + '</s>'
        input_ids = self.tokenizer.encode(text)
        text_len = len(input_ids)
        if text_len > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
        else:
            input_ids = input_ids + [0] * (self.max_seq_len - text_len)
        input_ids = np.array(input_ids)
        X = np.array(input_ids[:-1]).astype(np.int64)
        Y = np.array(input_ids[1:]).astype(np.int64)
        return {
            'input_ids': torch.from_numpy(X),
            'labels': torch.from_numpy(Y),
        }
        
class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()
            
    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self, index):
        line = self.data[index]
        line = json.loads(line)
        instruction_text = line['instruction']
        input_text = line['input']
        output_text = line['output']
        history = line['history']
        query = instruction_text + input_text
        answer = output_text + self.tokenizer.eos_token
        messages = []
        if history:
            for i in history:
                messages.append({'role': 'user', 'content': i[0]})
                messages.append({'role': 'assistant', 'content': i[1]})
        
        messages.append({'role': 'user', 'content': query})   
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 
        prompt_input_ids = self.tokenizer.encode(prompt)
        answer_input_ids = self.tokenizer.encode(answer)
        input_ids = prompt_input_ids + answer_input_ids
        labels = [0] * len(prompt_input_ids) + answer_input_ids
        text_len = len(input_ids)
        if text_len > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
        else:
            input_ids = input_ids + [0] * (self.max_seq_len - text_len)
            labels = labels + [0] * (self.max_seq_len - text_len)
        
        input_ids = input_ids[:-1]
        labels = labels[1:]
        return {'input_ids': torch.tensor(input_ids), 'labels': torch.tensor(labels)}
    
    
# 内存不够，可使用如下方法加载数据
# class LLMDataset(IterableDataset):
#     def __init__(self, data_path, tokenizer, max_seq_len):
#         super().__init__()
#         self.data_path = data_path
#         self.tokenizer = tokenizer
#         self.max_seq_len = max_seq_len
    
#     def __iter__(self):
#         return self.data_process()
    
#     def data_process(self):
#         with open(self.data_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 line = json.loads(line)
#                 text = '<s>' + line['text'] + '</s>'
#                 input_ids = self.tokenizer.encode(text)
#                 text_len = len(input_ids)
#                 if text_len > self.max_seq_len:
#                     input_ids = input_ids[:self.max_seq_len]
#                 else:
#                     input_ids = input_ids + [0] * (self.max_seq_len - text_len)
#                 input_ids = np.array(input_ids)
#                 X = np.array(input_ids[:-1]).astype(np.int64)
#                 Y = np.array(input_ids[1:]).astype(np.int64)
#                 yield {
#                     'input_ids': torch.from_numpy(X),
#                     'labels': torch.from_numpy(Y),
#                 }

import json
from torch.utils.data import Dataset

class DPODataset(Dataset):
    def __init__(self, data_path, tokenizer):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.datas = []

        with open(self.data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    
                    # 必须包含 chosen 和 rejected
                    if "chosen" not in data or "rejected" not in data:
                        print(f"Warning: Skipping line {i} (missing 'chosen' or 'rejected')")
                        continue
                    
                    # 必须是 list 类型
                    if not isinstance(data["chosen"], list) or not isinstance(data["rejected"], list):
                        print(f"Warning: Skipping line {i} ('chosen'/'rejected' not a list)")
                        continue
                    
                    # 至少要有两条消息（user + assistant）
                    if len(data["chosen"]) < 2 or len(data["rejected"]) < 2:
                        print(f"Warning: Skipping line {i} (less than 2 messages)")
                        continue
                    
                    # 最后一条必须是 assistant
                    if data["chosen"][-1].get("role") != "assistant" or data["rejected"][-1].get("role") != "assistant":
                        print(f"Warning: Skipping line {i} (last message not from 'assistant')")
                        continue
                    
                    # 可选：检查 prompt 部分是否一致（DPO 前提）
                    prompt_chosen = data["chosen"][:-1]
                    prompt_rejected = data["rejected"][:-1]
                    if prompt_chosen != prompt_rejected:
                        print(f"Warning: Line {i} has inconsistent prompt between chosen/rejected. Using 'chosen' as prompt.")
                        # 仍保留，但以 chosen 为准（或可选择跳过）

                    self.datas.append(data)

                except Exception as e:
                    print(f"Error parsing line {i}: {e}")

    def __getitem__(self, index):
        sample = self.datas[index]
        chosen_messages = sample["chosen"]
        rejected_messages = sample["rejected"]

        # 提取 prompt: 所有消息 except 最后一个 assistant 回复
        prompt_messages = chosen_messages[:-1]  # 使用 chosen 的上下文作为 prompt

        # 构造带模板的 prompt 字符串（含 generation prompt，如 <|im_start|>assistant）
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 提取 assistant 的回复内容
        chosen_response = chosen_messages[-1]["content"]
        rejected_response = rejected_messages[-1]["content"]

        # Tokenize 各部分（不加 special tokens，因为 apply_chat_template 已处理）
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        chosen_ids = self.tokenizer(chosen_response, add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]
        rejected_ids = self.tokenizer(rejected_response, add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]

        return [prompt_ids, chosen_ids, rejected_ids]

    def __len__(self):
        return len(self.datas)
    
    
class DPODataCollator:
    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    def __call__(self, features):
        inputs_ids = []
        labels = []
        
        for feature in features:
            inputs_ids.append(feature[0] + feature[1])
            labels.append([0]*len(feature[0]) + feature[1])
        for feature in features:
            inputs_ids.append(feature[0] + feature[2])
            labels.append([0]*len(feature[0]) + feature[2])
            
        def process(inputs_ids, labels):
            inputs_ids = [input_ids[:self.max_seq_len] for input_ids in inputs_ids]
            labels = [label[:self.max_seq_len] for label in labels]
            max_len = max([len(input_ids) for input_ids in inputs_ids])
            batch_input_ids = []
            batch_labels = []
            
            for input_ids, label in zip(inputs_ids, labels):
                if len(input_ids) <= max_len:
                    input_ids = input_ids+[0]*(max_len-len(input_ids))
                    label = label+[0]*(max_len-len(label))
                    batch_input_ids.append(input_ids[:-1])
                    batch_labels.append(label[1:])
            return batch_input_ids, batch_labels
        
        inputs_ids, labels = process(inputs_ids, labels)
        
        return {
            "input_ids": torch.tensor(inputs_ids),
            "labels": torch.tensor(labels)
            }
        
        
            
            
