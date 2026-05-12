"""
=== Step 1: 数据准备 ===

全链路位置: 最前端，为 SFT 和 DPO 准备训练数据
目的: 下载并格式化指令数据 + 偏好对数据

这一步没有 TODO — 数据处理不是核心知识点。
"""
import json
import random
from pathlib import Path
from datasets import load_dataset

random.seed(42)


def prepare_sft_data():
    """准备 SFT 指令微调数据"""
    output_dir = Path("data/sft")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("📥 Loading Alpaca-zh dataset...")
    ds = load_dataset("silk-road/alpaca-data-gpt4-chinese", split="train")

    samples = []
    for item in ds:
        instruction = item["instruction"]
        input_text = item.get("input", "")
        output = item["output"]

        user_content = f"{instruction}\n\n{input_text}" if input_text.strip() else instruction

        samples.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output},
            ]
        })

    random.shuffle(samples)
    train = samples[:5000]
    val = samples[5000:5500]

    for path, data in [(output_dir / "train.jsonl", train), (output_dir / "val.jsonl", val)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  ✅ {path}: {len(data)} samples")


def prepare_dpo_data():
    """准备 DPO 偏好对数据"""
    output_dir = Path("data/dpo")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("📥 Loading UltraFeedback preference dataset...")
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")

    samples = []
    for item in ds:
        try:
            prompt = item["chosen"][0]["content"]
            chosen = item["chosen"][1]["content"]
            rejected = item["rejected"][1]["content"]
            if len(prompt) > 10 and len(chosen) > 10 and len(rejected) > 10:
                samples.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        except (KeyError, IndexError):
            continue

    random.shuffle(samples)
    train = samples[:3000]
    val = samples[3000:3300]

    for path, data in [(output_dir / "train.jsonl", train), (output_dir / "val.jsonl", val)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  ✅ {path}: {len(data)} pairs")


if __name__ == "__main__":
    prepare_sft_data()
    print()
    prepare_dpo_data()
    print("\n🎯 Data ready. Next: python scripts/2_sft_train.py")
