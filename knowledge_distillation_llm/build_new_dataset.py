import json
import random

# 设置随机种子（可选，确保每次结果一致）
random.seed(42)

input_file = '/root/autodl-tmp/train_llm_from_scratch/sft_data_zh.jsonl'
output_file = './data_zh.json'
sample_count = 100000  # 需要抽取的样本数量

# 第一步：统计总行数（非空行）
print("正在统计总样本数...")
total_lines = 0
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            total_lines += 1

print(f"总有效样本数: {total_lines}")

if total_lines < sample_count:
    print(f"警告：总样本数 ({total_lines}) 少于目标数量 ({sample_count})，将使用全部样本。")
    sample_count = total_lines

# 第二步：随机选择要抽取的行索引
selected_indices = set(random.sample(range(total_lines), sample_count))
print(f"已随机选择 {len(selected_indices)} 个样本索引。")

# 第三步：再次遍历文件，提取选中的行
selected_data = []
current_idx = 0

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue  # 跳过空行
        
        if current_idx in selected_indices:
            try:
                data = json.loads(line)
                selected_data.append(data)
            except json.JSONDecodeError as e:
                print(f"警告：第 {current_idx + 1} 行 JSON 解析失败，已跳过。错误: {e}")
        
        current_idx += 1

print(f"成功抽取 {len(selected_data)} 条数据。")

# 第四步：保存为 JSON 数组格式
print(f"正在保存到 {output_file} ...")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(selected_data, f, ensure_ascii=False, indent=2)

print("完成！")