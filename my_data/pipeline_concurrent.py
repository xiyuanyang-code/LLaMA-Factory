import json
import random
from datasets import load_dataset
import openai
from tqdm import tqdm
import os
import concurrent.futures # 引入并发模块

# ==================== 配置 ====================
OUTPUT_FILE = "data/planning_coding.jsonl"
OPENAI_MODEL = "deepseek-v3"
MAX_WORKERS = 5 # 最大并发线程数，可以根据API速率限制和机器性能调整

def get_dataset():
    """加载数据集并返回编码任务列表。"""
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
    """调用OpenAI API生成带规划的代码，这是I/O密集型操作。"""
    prompt = PLANNING_PROMPT.format(task_description=task_description, GT=GT)
    # 注意：在实际部署中，您需要确保这里的base_url和api_key配置正确
    client = openai.OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:18889/v1")
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2048, # 增加max_tokens以确保能容纳规划和分步代码
        )
        # print(f"Getting Response: {response.choices[0].message.content.strip()}") # 并发模式下频繁打印会干扰进度条
        return response.choices[0].message.content.strip()
    except Exception as e:
        # 记录错误，防止程序中断
        print(f"[ERROR] API call failed: {e}")
        return f"[ERROR] {e}"

def process_task(task: dict) -> dict:
    """单个任务的处理器，用于在线程池中执行。"""
    full_response = generate_planning_code(
        task_description=task["prompt"],
        GT=task["completion"]
    )
    # 构造最终要写入JSONL的字典项
    return {"instruction": task, "input": "", "output": full_response}


def main():
    """主函数，负责加载数据、设置并发执行和结果收集。"""
    print("Loading Datasets...")
    coding_tasks = get_dataset()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # 计数器，记录成功写入的文件数量
    tasks_completed = 0

    print(f"Starting concurrent data generation with {MAX_WORKERS} workers...")
    
    # 在并发执行之前打开文件，并将写入操作放在主线程中，以确保线程安全。
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        # 使用 ThreadPoolExecutor 实现并发
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交所有任务到线程池
            future_to_task = {executor.submit(process_task, task): task for task in coding_tasks}
            
            # 使用 tqdm 包装 concurrent.futures.as_completed 来显示进度条
            for future in tqdm(
                concurrent.futures.as_completed(future_to_task), 
                total=len(coding_tasks), 
                desc="Generating planning data (Writing on completion)"
            ):
                try:
                    # 获取任务结果
                    result_item = future.result()
                    
                    # 立即写入文件并刷新缓冲区
                    f.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                    f.flush() # 强制写入磁盘
                    tasks_completed += 1
                except Exception as exc:
                    # 捕获并打印任务执行过程中发生的任何异常
                    task_failed = future_to_task[future]
                    print(f'\n[Task Error] Task for prompt "{task_failed.get("prompt", "Unknown")[:30]}..." generated an exception: {exc}')
                    # 任务失败则跳过写入，但进度条继续
    
    # 当所有任务完成后，文件 f 会被自动关闭
    print(f"\n生成完成！成功写入 {tasks_completed} 个任务数据至：{OUTPUT_FILE}")


def test_calling():
    # 保持测试函数不变
    client = openai.OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:18889/v1")
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": "Introduce yourself"}],
        temperature=0.7,
        max_tokens=1024,
    )
    print(response)

def get_file_data(line: str):
    line_data = json.loads(line)
    return {
        "instruction": "Solving the problems below, remember to draft a planning in chinese, and then finish your sub-tasks step-by-step",
        "input": line_data["instruction"]["prompt"],
        "output": line_data["output"]
    }

def get_json_file_path():
    file_path = "./data/planning_coding.jsonl"
    with open(file_path, "r", encoding="utf-8") as file:
        data = [get_file_data(line=line) for line in file]
    with open("./data/planning_coding.json", "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    

if __name__ == "__main__":
    # main()
    get_json_file_path()