import os
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='X Y correlation')
    parser.add_argument("--x", type=str, default="single_influence_score", help="X")
    parser.add_argument("--y", type=str, default="policy_loss", help="Y")
    args = parser.parse_args()
    return args

def read_jsonl(file_path):
    data_list = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)  # 将每行转换为字典
            data_list.append(data)
    return data_list

def get_train_info():
    train_info = []
    for i in range(4):
        status_path = f"../tinyllamachat_global_step10_gradients_train/tinyllamachat_global_step10_gradients_train_part_0-3/tinyllamachat_global_step10_gradients_train_part_{i}/status/status.jsonl"
        status = read_jsonl(status_path)
        train_info.extend(status)
    for i in range(4):
        status_path = f"../tinyllamachat_global_step10_gradients_train/tinyllamachat_global_step10_gradients_train_part_4-7/tinyllamachat_global_step10_gradients_train_part_{i+4}/status/status.jsonl"
        status = read_jsonl(status_path)
        train_info.extend(status)
    for i in range(2):
        status_path = f"../tinyllamachat_global_step10_gradients_train/tinyllamachat_global_step10_gradients_train_part_8-9/tinyllamachat_global_step10_gradients_train_part_{i+8}/status/status.jsonl"
        status = read_jsonl(status_path)
        train_info.extend(status)
    return train_info

def get_influence():
    folder_path = '../influence'
    all_scores = []
    for i in range(200):
        file_path = os.path.join(folder_path, f'scores_{i}')
        with open(file_path, 'r') as f:
            scores = json.load(f)
            all_scores.append(scores)
    return all_scores #200*4w


def get_eval_info():
    status_path = f"../tinyllamachat_global_step10_gradients_evaluation_saferlhf/status/status.jsonl"
    eval_status = read_jsonl(status_path)
    return eval_status

if __name__ == '__main__':
    args = parse_arguments()
    train_info = get_train_info()
    eval_info = get_eval_info()
    influence_scores = get_influence()

    # mean_influencec_scores = np.mean(influence_scores, axis=0)
    single_influence_scores = influence_scores[12]

    # for info, mean_influence_score in zip(train_info, mean_influencec_scores):
    #     info['mean_influence'] = mean_influence_score

    for info, single_influence_score in zip(train_info, single_influence_scores):
        info['single_influence_score'] = single_influence_score

    sorted_list = sorted(train_info, key=lambda x: x['single_influence_score'], reverse=True)

    x = [data[f'{args.x}'] for data in sorted_list]
    y = [data[f'{args.y}'] for data in sorted_list]

    count = 0
    for i in x:
        if i > 0:
            count += 1
    print(count)
    print(len(x))
    plt.scatter(x, y, s=3)

    plt.title('Scatter Plot')
    plt.xlabel(f'X_{args.x}')
    plt.ylabel(f'Y_{args.y}')
    plt.savefig(f'../tmp/single_data_3_{args.x}-{args.y}.png')