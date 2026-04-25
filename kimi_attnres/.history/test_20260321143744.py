from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from train import LLM, Config
t = AutoTokenizer.from_pretrained('./sft/checkpoint-12000')
AutoConfig.register("moe_model", Config)
AutoModelForCausalLM.register(Config, LLM)
model = AutoModelForCausalLM.from_pretrained('./sft/checkpoint-12000')

input_data = t.apply_chat_template([{'role':'user', 'content':'讲一个故事'}],add_generation_prompt=True)
# input_data = [t.bos_token_id] + t.encode('你是一个')
print(input_data)

for token in model.generate({"input_ids":torch.tensor(input_data).unsqueeze(0), "labels":None}, t.eos_token_id, 200, stream=False,temperature=0.0, top_k=1):
    print(t.decode(token[0]))