# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# model = AutoModelForCausalLM.from_pretrained('yifangong/alpaca-1.1b-reproduced-tinyllama-v1.1', torch_dtype=torch.bfloat16, device_map='auto')
# tokenizer = AutoTokenizer.from_pretrained('yifangong/alpaca-1.1b-reproduced-tinyllama-v1.1')

# while True:
#     template = "User: {}\nAssistant: "
#     prompt = input()
#     prompt = template.format(prompt)
#     input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()
#     output_ids = model.generate(input_ids, max_new_tokens=128)[0]

#     print(tokenizer.decode(output_ids, skip_special_tokens=True))


# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
while True:
    content = input('input: ')
    messages = [
        {"role": "user", "content": content},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    print(outputs[0]["generated_text"])
