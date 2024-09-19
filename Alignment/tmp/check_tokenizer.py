from transformers import AutoTokenizer

# 加载预训练的 tokenizer（替换成你自己的模型名）
tokenizer = AutoTokenizer.from_pretrained("PKU-Alignment/alpaca-7b-reproduced-llama-2")

# 获取 tokenizer 的词表大小
vocab_size = tokenizer.vocab_size

print(f"Tokenizer 词表大小: {vocab_size}")

tokenizer = AutoTokenizer.from_pretrained("OpenRLHF/Llama-2-7b-rm-anthropic_hh-lmsys-oasst-webgpt")

# 获取 tokenizer 的词表大小
vocab_size = tokenizer.vocab_size

print(f"Tokenizer 词表大小: {vocab_size}")