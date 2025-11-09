from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


def run_message(model_path: str, message: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # 自动分配到 GPU
        trust_remote_code=True,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=2048,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    messages = [
        {"role": "system", "content": "You are en expert in coding, your task is Solving the problems below, remember to draft a planning in chinese, and then finish your sub-tasks step-by-step"},
        {"role": "user", "content": message},
    ]
    output = pipe(messages)
    print("\n\n" + "=" * 25)
    print(f"Message from {model_path}")
    print(output[0]["generated_text"])
    print("\n\n" + "=" * 25 + "\n\n")


if __name__ == "__main__":
    message = "Help me use Python to generate a random digit number with any given length."
    model_path_list = [
        "/home/ma-user/work/xiyuanyang/LLaMA-Factory/models/qwen2_5_3B-sft-planning",
        "Qwen/Qwen2.5-3B",
    ]

    for model_path in model_path_list:
        run_message(model_path=model_path,message=message)
