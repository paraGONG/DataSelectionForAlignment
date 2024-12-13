import json
import pandas as pd
import argparse

# 读取jsonl文件并提取reward值



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_num", type=int, default=0)
    parser.add_argument("--select_policy", type=str, default=None)

    args = parser.parse_args()

    rewards = []
    with open(f"../buffer/window_{args.window_num}/device_0/output/output.jsonl", 'r') as f:
        for line in f:
            data = json.loads(line)
            if 'reward' in data:
                rewards.append(data['reward'])

    # 计算均值
    mean_reward = pd.Series(rewards).mean()

    print(f"均值: {mean_reward}")
