from datasets import load_dataset

# 加载数据集
dataset = load_dataset("OpenRLHF/prompt-collection-v0.1")

# 查看数据集的列
print(dataset.column_names)

# 遍历 context_messages 列
for i,data in enumerate(dataset):
    context_messages = data['context_messages']
    print(context_messages)
    if i == 10:
        break