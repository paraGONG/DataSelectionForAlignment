import json

def save_to_jsonl(string, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for i in range(32):
            json_obj = {"prompt": string}
            f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

# 示例字符串列表
string = "Can I get multiple free samples at a store?"

# 保存到JSONL文件
save_to_jsonl(string, 'eval_prompt_0.jsonl')
