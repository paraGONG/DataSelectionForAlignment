from datasets import load_dataset
import json

dataset_path = 'stingning/ultrachat'
dataset = load_dataset(dataset_path)
dataset_name = dataset_path.spilt('/')[-1]

output_file = f'../my_dataset/{dataset_name}.jsonl'

with open(output_file, 'w', encoding='utf-8') as f:
    for row in dataset:
        data = {
            "prompt": row['data'][0],
            "source": dataset_name
        }
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
