import json

# 输入文件路径
input_file = '../../data/helpful_test/helpful.jsonl'

# 读取原始JSONL文件，处理数据
with open(input_file, 'r', encoding='utf-8') as infile:
    lines = infile.readlines()

# 打开文件进行覆盖写入
with open(input_file, 'w', encoding='utf-8') as outfile:
    for line in lines:
        # 解析每一行的JSON数据
        data = json.loads(line)
        
        # 提取Human说的话
        chosen_text = data.get('chosen', '')
        human_prompt = chosen_text.split('\n\nHuman: ')[-1].split('\n\nAssistant:')[0].strip()
        
        # 创建新的字典，使用prompt作为键
        new_data = {'prompt': human_prompt}
        
        # 写入新的字典到文件
        outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')

print(f"已成功覆盖原文件 {input_file}")
