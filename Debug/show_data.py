import os
import json
import numpy as np
import matplotlib.pyplot as plt


def read_jsonl(file_path):
    data_list = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)  # 将每行转换为字典
            data_list.append(data)
    return data_list

def get_train_output():
    train_output = []
    for i in range(4):
        output_path = f"../tinyllamachat_global_step10_gradients_train/tinyllamachat_global_step10_gradients_train_part_0-3/tinyllamachat_global_step10_gradients_train_part_{i}/output/output.jsonl"
        output = read_jsonl(output_path)
        train_output.extend(output)
    for i in range(4):
        output_path = f"../tinyllamachat_global_step10_gradients_train/tinyllamachat_global_step10_gradients_train_part_4-7/tinyllamachat_global_step10_gradients_train_part_{i+4}/output/output.jsonl"
        output = read_jsonl(output_path)
        train_output.extend(output)
    for i in range(2):
        output_path = f"../tinyllamachat_global_step10_gradients_train/tinyllamachat_global_step10_gradients_train_part_8-9/tinyllamachat_global_step10_gradients_train_part_{i+8}/output/output.jsonl"
        output = read_jsonl(output_path)
        train_output.extend(output)
    return train_output[:100]

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
    return train_info[:100]

def get_influence():
    folder_path = '../debug/self_influence'
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
    train_info = get_train_info()
    influence_scores = get_influence()
    # for i in range(100):
    #     single_influence_scores = influence_scores[i]

    # for info, single_influence_score in zip(train_info, single_influence_scores):
    #     info['single_influence_score'] = single_influence_score

    # sorted_list = sorted(train_info, key=lambda x: x['single_influence_score'], reverse=True)

    plt.imshow(influence_scores, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Self Influence')
    plt.savefig(f'../tmp/self-influence.png')

    # mean_values = np.mean(all_scores, axis=0)
    # k = 10240
    # top_k_indices = np.argsort(mean_values)[-k:][::-1]

    # with open("../candidate_dataset.jsonl", 'r') as f:
    #     data = [json.loads(line) for line in f]
    # selected_data = [data[i] for i in top_k_indices]

    # with open("../selected_dataset.jsonl", 'w') as f:
    #     for item in selected_data:
    #         f.write(json.dumps(item) + '\n')