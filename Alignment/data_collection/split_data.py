import json
import os

os.makedirs('/candidate_dataset', exist_ok=True)

# 文件名和分割份数
input_file = 'condidate_dataset.jsonl'
num_parts = 10

# 读取输入文件
with open(input_file, 'r') as f:
    lines = f.readlines()

# 计算每一份的行数
total_lines = len(lines)
lines_per_part = total_lines // num_parts

# 分割并保存文件
for i in range(num_parts):
    start_index = i * lines_per_part
    # 如果是最后一部分，确保包含所有剩余行
    end_index = start_index + lines_per_part if i < num_parts - 1 else total_lines
    
    part_lines = lines[start_index:end_index]
    
    output_file = f'./candidate_dataset/data_{i}.jsonl'
    with open(output_file, 'w') as out_f:
        out_f.writelines(part_lines)

print("分割完成！")
