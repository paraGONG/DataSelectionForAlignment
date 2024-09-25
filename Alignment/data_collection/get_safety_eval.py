import json
import random

# 读取b.jsonl中的数据
with open('rlhf_prompt_collection_5w.jsonl', 'r', encoding='utf-8') as b_file:
    b_data = {json.loads(line)['prompt'] for line in b_file}  # 假设每条数据都有'id'字段

# 从a.jsonl中随机挑选数据
with open('./my_dataset/PKU-SafeRLHF.jsonl', 'r', encoding='utf-8') as a_file:
    a_data = [json.loads(line) for line in a_file if json.loads(line)['prompt'] not in b_data]

# 随机选择100条数据
sample_size = 200
sampled_data = random.sample(a_data, min(sample_size, len(a_data)))

# 将结果写入c.jsonl
with open('safetyrlhf_evaluation_dataset.jsonl', 'w', encoding='utf-8') as c_file:
    for item in sampled_data:
        c_file.write(json.dumps(item) + '\n')
