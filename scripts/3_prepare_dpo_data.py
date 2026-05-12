"""
Step 3: Prepare DPO preference data
策略：用 SFT 模型生成多个回答，根据质量构建 chosen/rejected pairs
也可以直接用开源偏好数据集
"""
import json
import random
from pathlib import Path
from datasets import load_dataset

OUTPUT_DIR = Path("data/dpo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def prepare_from_open_source():
    """
    使用开源偏好数据集
    可选数据源:
    1. HuggingFaceH4/ultrafeedback_binarized — 英文，高质量
    2. argilla/dpo-mix-7k — 混合
    3. 自己用 SFT 模型 + GPT-4 评分生成
    """
    print("📥 Loading preference dataset...")
    
    # 使用 ultrafeedback（高质量，英文为主）
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    
    print(f"   Total samples: {len(ds)}")
    
    samples = []
    for item in ds:
        try:
            prompt = item["chosen"][0]["content"]  # user message
            chosen = item["chosen"][1]["content"]   # preferred response
            rejected = item["rejected"][1]["content"]  # rejected response
            
            if len(prompt) > 10 and len(chosen) > 10 and len(rejected) > 10:
                samples.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                })
        except (KeyError, IndexError):
            continue
    
    print(f"   Valid pairs: {len(samples)}")
    
    # 取子集
    random.seed(42)
    random.shuffle(samples)
    n_train = min(3000, int(len(samples) * 0.9))
    n_eval = min(300, len(samples) - n_train)
    
    train_samples = samples[:n_train]
    eval_samples = samples[n_train:n_train + n_eval]
    
    for path, data in [
        (OUTPUT_DIR / "train.jsonl", train_samples),
        (OUTPUT_DIR / "eval.jsonl", eval_samples),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"✅ Saved {len(data)} pairs to {path}")

def prepare_self_generated(sft_model_path: str = "outputs/sft/final"):
    """
    进阶方案：用 SFT 模型自己生成 + 评分
    面试加分项：展示你能构建偏好数据
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"📥 Loading SFT model from {sft_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        sft_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    
    # 加载 SFT 训练用的 prompts
    prompts = []
    with open("data/sft/train.jsonl") as f:
        for line in f:
            item = json.loads(line)
            prompts.append(item["messages"][0]["content"])
    
    random.seed(42)
    prompts = random.sample(prompts, min(1000, len(prompts)))
    
    samples = []
    for i, prompt in enumerate(prompts):
        if i % 100 == 0:
            print(f"   Generating {i}/{len(prompts)}...")
        
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # 生成两个不同温度的回答
        outputs_good = model.generate(**inputs, max_new_tokens=256, temperature=0.3, do_sample=True)
        outputs_bad = model.generate(**inputs, max_new_tokens=256, temperature=1.2, do_sample=True)
        
        chosen = tokenizer.decode(outputs_good[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        rejected = tokenizer.decode(outputs_bad[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        if chosen != rejected and len(chosen) > 10:
            samples.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    
    # 保存
    random.shuffle(samples)
    n_train = int(len(samples) * 0.9)
    for path, data in [
        (OUTPUT_DIR / "train_selfgen.jsonl", samples[:n_train]),
        (OUTPUT_DIR / "eval_selfgen.jsonl", samples[n_train:]),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"✅ Saved {len(data)} self-generated pairs to {path}")

if __name__ == "__main__":
    prepare_from_open_source()
    # 可选：跑完 SFT 后再跑 self-generated
    # prepare_self_generated()
    print("\n🎯 DPO data ready! Next: python scripts/4_dpo_train.py")
