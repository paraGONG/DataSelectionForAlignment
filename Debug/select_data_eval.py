import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='debug')
    parser.add_argument("--index", type=int, default=0, help="index of eval data")
    parser.add_argument("--reverse", action="store_true", default=False, help="reverse")
    args = parser.parse_args()
    return args


def read_jsonl(file_path):
    data_list = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)  # 将每行转换为字典
            data_list.append(data)
    return data_list

def get_train_output():
    train_output = []
    for i in range(1):
        output_path = f"../tinyllamachat_global_step10_gradients_train/tinyllamachat_global_step10_gradients_train_part_0-3/tinyllamachat_global_step10_gradients_train_part_{i}/output/output.jsonl"
        output = read_jsonl(output_path)
        train_output.extend(output[:2000])
    for i in range(1):
        output_path = f"../tinyllamachat_global_step10_gradients_eval_prompt_0/output/output.jsonl"
        output = read_jsonl(output_path)
        train_output.extend(output)
    return train_output


def get_eval_output():
    output_path = "../tinyllamachat_global_step10_gradients_evaluation_saferlhf/output/output.jsonl"
    output = read_jsonl(output_path)
    return output


def get_train_info():
    train_info = []
    for i in range(1):
        status_path = f"../tinyllamachat_global_step10_gradients_train/tinyllamachat_global_step10_gradients_train_part_0-3/tinyllamachat_global_step10_gradients_train_part_{i}/status/status.jsonl"
        status = read_jsonl(status_path)
        train_info.extend(status[:2000])
    for i in range(1):
        status_path = f"../tinyllamachat_global_step10_gradients_eval_prompt_0/status/status.jsonl"
        status = read_jsonl(status_path)
        train_info.extend(status)
    return train_info


def get_influence():
    folder_path = '../eval_influence_0'
    all_scores = []
    for i in range(1):
        file_path = os.path.join(folder_path, f'scores_{i}')
        with open(file_path, 'r') as f:
            scores = json.load(f)
            all_scores.append(scores)
    return all_scores


def get_influence_cosine():
    folder_path = '../debug/self_influence_cosine'
    all_scores = []
    for i in range(100):
        file_path = os.path.join(folder_path, f'scores_{i}')
        with open(file_path, 'r') as f:
            scores = json.load(f)
            all_scores.append(scores)
    return all_scores


def get_eval_info():
    status_path = f"../tinyllamachat_global_step10_gradients_evaluation_saferlhf/status/status.jsonl"
    eval_status = read_jsonl(status_path)
    return eval_status


if __name__ == '__main__':
    args = parse_arguments()
    train_info = get_train_info()
    train_output = get_train_output()
    eval_info = get_eval_info()
    eval_output = get_eval_output()
    influence_scores = get_influence()

    index = args.index
    target_scores = influence_scores[0]
    for info, score, output in zip(train_info, target_scores, train_output):
        info['influence_score'] = score
        info['output'] = output['output']
    sorted_list = []

    if args.reverse:
            sorted_list = sorted(train_info, key=lambda x: x['influence_score'], reverse=False)
    else:
        sorted_list = sorted(train_info, key=lambda x: x['influence_score'], reverse=True)

    # plt.imshow(influence_scores, cmap='hot', interpolation='nearest')
    # plt.colorbar()
    # plt.title('Self Influence Cosine')
    # plt.savefig(f'../tmp/self-influence-cosine.png')
    print(eval_output[index]['output'])
    print(eval_info[index])
    save_path = f"../debug_eval/selected_data_eval_prompt_0"
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f"eval_num_{index}_reverse_{args.reverse}.jsonl"), 'w') as f:
        for item in sorted_list:
            f.write(json.dumps(item) + '\n')