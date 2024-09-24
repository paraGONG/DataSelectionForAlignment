import pandas as pd

# 读取jsonl文件
data = pd.read_json('../rlhf_prompt_collection_5w.jsonl', lines=True)

sampled_data = data.sample(n=10240, random_state=42)

sampled_data.to_json('../warmup_dataset.jsonl', orient='records', lines=True)

# 保存剩余数据
remaining_data = data.drop(sampled_data.index)
remaining_data.to_json('../candidata_dataset.jsonl', orient='records', lines=True)
