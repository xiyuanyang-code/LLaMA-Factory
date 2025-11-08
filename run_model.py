from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_path = "/home/ma-user/work/xiyuanyang/LLaMA-Factory/models/qwen2_5_3B-sft"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # 自动分配到 GPU
    trust_remote_code=True
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=2048,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "写一首关于秋天的诗"}
]

output = pipe(messages)
print(output[0]["generated_text"])