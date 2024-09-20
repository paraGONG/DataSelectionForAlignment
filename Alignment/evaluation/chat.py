import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('yifangong/alpaca-1.1b-reproduced-tinyllama-v1.1', torch_dtype=torch.bfloat16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('yifangong/alpaca-1.1b-reproduced-tinyllama-v1.1')

while True:
    template = "User: {}\nAssistant: "
    prompt = input()
    prompt = template.format(prompt)
    input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()
    output_ids = model.generate(input_ids, max_new_tokens=128)[0]

    print(tokenizer.decode(output_ids, skip_special_tokens=True))
