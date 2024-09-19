from datasets import load_dataset
import json

# 加载数据集
dataset = load_dataset("OpenRLHF/prompt-collection-v0.1")

# 查看数据集的列
print(dataset['train'].column_names)

count = 0
# 遍历 context_messages 列
for i,data in enumerate(dataset['train']):
    context_messages = data['context_messages']
    if len(context_messages) > 1:
        count += 1
        print(context_messages)
        print("\n")
print(count)