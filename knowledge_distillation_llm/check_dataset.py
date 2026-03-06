import json

file_path = '/root/autodl-tmp/train_llm_from_scratch/sft_data_zh.jsonl'

total_lines = 0
valid_samples = 0

print("=== 前5条有效数据 ===")
with open(file_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        total_lines += 1
        line = line.strip()
        if not line:
            continue  # 跳过空行
        
        try:
            data = json.loads(line)
            valid_samples += 1
            
            # 打印前5条有效样本
            if valid_samples <= 5:
                print(f"--- 样本 {valid_samples} (文件第 {i+1} 行) ---")
                print(json.dumps(data, ensure_ascii=False, indent=2))
                
        except json.JSONDecodeError:
            print(f"警告：第 {i+1} 行不是合法 JSON，已跳过")

print("\n=== 统计信息 ===")
print(f"文件总行数（含空行/无效行）: {total_lines}")
print(f"有效 JSONL 样本数量: {valid_samples}")