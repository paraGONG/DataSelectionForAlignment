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
print(mean_values)
# k = 3
# top_k_indices = np.argsort(list(means.values()))[-k:]

# top_k_files = {list(means.keys())[i]: means[list(means.keys())[i]] for i in top_k_indices}
# print(top_k_files)
