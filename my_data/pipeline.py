# generate_planning_data.py
import json
import random
from datasets import load_dataset
import openai
from tqdm import tqdm
import os

# ==================== 配置 ====================
OUTPUT_FILE = "data/planning_coding.jsonl"
OPENAI_MODEL = "deepseek-v3"

def get_dataset():
    ds = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train")
    coding_tasks = [x for x in ds]
    print(f"The length of the coding tasks: {len(coding_tasks)}")
    return coding_tasks

# 2. Planning 引导 Prompt
PLANNING_PROMPT = """你是一个高级算法工程师。真正优秀的算法工程师总会在写代码之前做好任务的规划，因此，请你在解决这些问题之前**显式地做出行之有效的任务规划**，并说“真正的大师总是会在规划之后再动手”。

你需要解决的问题是：
{task_description}

你可以看到标准答案，但是**请你根据你的规划逐步生成代码**，你必须输出你的一步步完成规划的轨迹！最关键的不是答案，而是你的规划能力！

{GT}

请严格按照以下格式回答：

真正的大师总是需要规划之后再动手！我先做一个规划：
### Planning
1. ...
2. ...
3. ...

### Code

#### Solving Sub-Tasks1
```the programming language
...
```

#### Solving Sub-Tasks2
```the programming language
...
```

#### Solving Sub-Tasks3
```the programming language
...
```

#### Solving Sub-Tasks4
```the programming language
...
```

...

#### Final Code
```the programming language
...
```

"""


def generate_planning_code(task_description: str, GT: str) -> str:
    prompt = PLANNING_PROMPT.format(task_description=task_description, GT=GT)
    client = openai.OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:18889/v1")
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
        )
        print(f"Getting Response: {response.choices[0].message.content.strip()}")
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] {e}"


def main():
    print("Loading Datasets...")
    coding_tasks = get_dataset()
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for task in tqdm(coding_tasks, total=len(coding_tasks)):
            full_response = generate_planning_code(
                task_description=task["prompt"],
                GT=task["completion"]
            ) 
            item = {"instruction": task, "input": "", "output": full_response}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"生成完成！数据保存至：{OUTPUT_FILE}")


def test_calling():
    client = openai.OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:18889/v1")
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": "Introduce yourself"}],
        temperature=0.7,
        max_tokens=1024,
    )
    print(response)

if __name__ == "__main__":
    main()

