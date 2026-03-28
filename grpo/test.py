import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = './output/checkpoint_200/'
SYSTEM_PROMPT = "你是数学小助手，需要按照指定格式输出答案。请首先在 Thinking 标签中展示你的思考过程，然后在 Answer 标签中给出最终答案。格式如下：\n<think>\n你的思考过程\n</think>\n<answer>\n你的答案\n</answer>"

def main():
    print("正在加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("模型加载完成！")

    while True:
        user_input = input("\n请输入问题（或输入 q 退出）: ")
        if user_input.lower() == 'q':
            break

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]
        
        input_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        inputs = tokenizer(
            [input_text], 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"\n模型输出:\n{response}")

if __name__ == "__main__":
    main()
