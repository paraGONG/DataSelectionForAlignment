import json
import pandas as pd

# 读取jsonl文件并提取reward值
rewards = []
with open('./eval_results/saferlhf_evaluation_dataset_tinyllama_random_TinyLlama-1.1B-Chat-v1.0-reward-model.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if 'reward' in data:
            rewards.append(data['reward'])

# 计算均值
mean_reward = pd.Series(rewards).mean()

print(f"均值: {mean_reward}")
