from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "PKU-Alignment/alpaca-7b-reproduced-llama-2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForCausalLM.from_pretrained(checkpoint)  # You may want to use bfloat16 and/or move to GPU here

messages = [
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
 ]
# tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
# print(tokenizer.decode(tokenized_chat[0]))
print(tokenizer.default_chat_template)