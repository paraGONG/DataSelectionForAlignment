import json
import random

# 定义输入和输出文件路径
input_file = '/root/siton-object-46b8630eb56e449886cb89943ab6fe10/dataset/PKU-SafeRLHF-prompt/data/PKU-SafeRLHF-prompts.jsonl'
output_file = '/root/siton-object-46b8630eb56e449886cb89943ab6fe10/dataset/PKU-SafeRLHF-prompt/data/PKU-SafeRLHF-prompts-random-sample-50.jsonl'

# 读取jsonl文件中的所有行
with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 随机选择50行
selected_lines = random.sample(lines, 50)

# 将选中的行写入新的jsonl文件
with open(output_file, 'w', encoding='utf-8') as f:
    for line in selected_lines:
        f.write(line)

print(f'Successfully saved 50 randomly selected lines to {output_file}')
