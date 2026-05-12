"""
Step 1: Prepare SFT training data
下载 Alpaca-zh 中文指令数据集，转换为 messages 格式
"""
import json
import random
from pathlib import Path
from datasets import load_dataset

OUTPUT_DIR = Path("data/sft")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def prepare_alpaca_zh():
    """下载 Alpaca-zh 并转换格式"""
    print("📥 Loading Alpaca-zh dataset...")
    ds = load_dataset("silk-road/alpaca-data-gpt4-chinese", split="train")
    
    print(f"   Total samples: {len(ds)}")
    
    # 转换为 messages 格式
    samples = []
    for item in ds:
        instruction = item["instruction"]
        input_text = item.get("input", "")
        output = item["output"]
        
        # 合并 instruction 和 input
        if input_text and input_text.strip():
            user_content = f"{instruction}\n\n{input_text}"
        else:
            user_content = instruction
        
        samples.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output}
            ]
        })
    
    # 打乱并切分
    random.seed(42)
    random.shuffle(samples)
    
    # 取 5000 条训练 + 500 条验证（小模型不需要太多数据）
    n_train = min(5000, int(len(samples) * 0.9))
    n_eval = min(500, len(samples) - n_train)
    
    train_samples = samples[:n_train]
    eval_samples = samples[n_train:n_train + n_eval]
    
    # 写入 JSONL
    train_path = OUTPUT_DIR / "train.jsonl"
    eval_path = OUTPUT_DIR / "eval.jsonl"
    
    for path, data in [(train_path, train_samples), (eval_path, eval_samples)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"✅ Saved {len(data)} samples to {path}")
    
    # 打印样例
    print("\n📝 Sample data:")
    sample = train_samples[0]
    print(f"   User: {sample['messages'][0]['content'][:100]}...")
    print(f"   Assistant: {sample['messages'][1]['content'][:100]}...")

if __name__ == "__main__":
    prepare_alpaca_zh()
    print("\n🎯 SFT data ready! Next: python scripts/2_sft_train.py")
