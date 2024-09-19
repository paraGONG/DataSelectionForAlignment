from transformers import AutoTokenizer

# 加载预训练的 tokenizer（替换成你自己的模型名）
tokenizer = AutoTokenizer.from_pretrained("PKU-Alignment/alpaca-7b-reproduced-llama-2")

# 获取 tokenizer 的词表大小
vocab_size = tokenizer.vocab_size

print(f"Tokenizer 词表大小: {vocab_size}")
