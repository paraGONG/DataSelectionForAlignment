import os
import json
import numpy as np

folder_path = '../influence'
means = {}

all_scores = []
for i in range(200):
    file_path = os.path.join(folder_path, f'scores_{i}')
    with open(file_path, 'r') as f:
        scores = json.load(f)
        all_scores.append(scores)

mean_values = np.mean(all_scores, axis=0)
k = 10240
top_k_indices = np.argsort(mean_values)[-k:][::-1]

with open("../candidate_dataset.jsonl", 'r') as f:
    data = [json.loads(line) for line in f]
selected_data = [data[i] for i in top_k_indices]

with open("../selected_dataset.jsonl", 'w') as f:
    for item in selected_data:
        f.write(json.dumps(item) + '\n')