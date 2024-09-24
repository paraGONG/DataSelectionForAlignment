import json
import random

input_file = 'candidate_dataset.jsonl'
selected_file = 'random_dataset.jsonl'
# remaining_file = 'candidate_dataset.jsonl'
num_samples = 10240

# 读取原始数据
with open(input_file, 'r') as f:
    data = [json.loads(line) for line in f]

# 随机挑选数据
selected_data = random.sample(data, min(num_samples, len(data)))
# remaining_data = [item for item in data if item not in selected_data]

# 保存选中的数据
with open(selected_file, 'w') as f:
    for item in selected_data:
        f.write(json.dumps(item) + '\n')

# 保存剩余的数据
# with open(remaining_file, 'w') as f:
#     for item in remaining_data:
#         f.write(json.dumps(item) + '\n')
