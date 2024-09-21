import os
import json
from datasets import load_dataset

# dataset_path = 'stingning/ultrachat'
dataset_path = 'nvidia/HelpSteer'
dataset = load_dataset(dataset_path)
dataset_name = dataset_path.split('/')[-1]

output_file = f'../my_dataset/{dataset_name}-validation.jsonl'
os.makedirs('../my_dataset', exist_ok=True)

seen_questions = set()

with open(output_file, 'w', encoding='utf-8') as f:
    for row in dataset['validation']:
        prompt = row['prompt']
        if prompt not in seen_questions:
            seen_questions.add(prompt)
            data = {
                "prompt": prompt,
                "source": dataset_name
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
