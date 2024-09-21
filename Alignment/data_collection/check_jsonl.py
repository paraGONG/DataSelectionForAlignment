import json

jsonl_file = '../my_dataset/hh-rlhf-train.jsonl'

data_count = 0
first_ten_rows = []

with open(jsonl_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        data_count += 1
        
        if data_count <= 10:
            first_ten_rows.append(data)


# 打印前10条数据
print("前10条数据:")
for idx, row in enumerate(first_ten_rows, 1):
    print(f"第 {idx} 条: {json.dumps(row, ensure_ascii=False, indent=4)}")

# 打印统计结果
print(f"总共有 {data_count} 条数据。")