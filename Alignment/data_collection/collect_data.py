import os
import json
import random

def read_jsonl(file_path, num_samples):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    if len(data) <= num_samples:
        return data
    else:
        return random.sample(data, num_samples)

def merge_and_shuffle_jsonl(files_with_samples, output_file):
    all_data = []
    seen = set()
    # 从每个文件中读取指定数量的数据
    for file_path, num_samples in files_with_samples:
        all_data.extend(read_jsonl(file_path, num_samples))
    
    # 打乱数据顺序
    random.shuffle(all_data)

    # 将结果保存到新的 JSONL 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_data:
            prompt = item['prompt']
            if prompt not in seen:
                seen.add(prompt)
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"合并后的数据已保存到 {output_file}")

# 示例：指定每个文件和从中提取的样本数
files_with_samples = [
    ('./my_dataset/ultrachat.jsonl', 30000),  # 从 file1.jsonl 中随机挑选 100 条数据
    ('./my_dataset/HelpSteer-trai.jsonl', 10000),  # 从 file2.jsonl 中随机挑选 150 条数据
    ('./my_dataset/UltraInteract_pair.jsonl', 20000),
    ('./my_dataset/hh-rlhf-train.jsonl', 20000),
    ('./my_dataset/PKU-SafeRLHF.jsonl', 20000),
]

os.makedirs('../rlhf_prompt_collection_v1.0', exist_ok=True)

# 输出文件路径
output_file = '../rlhf_prompt_collection_v1.0/data.jsonl'

# 执行合并和随机打乱操作
merge_and_shuffle_jsonl(files_with_samples, output_file)
