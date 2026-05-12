"""
Step 2: SFT Training with LoRA
Qwen3-0.6B + LoRA on Chinese instruction data
"""
import yaml
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

def load_config():
    with open("configs/sft_config.yaml") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    model_name = cfg["model"]["name"]
    
    print(f"🚀 Loading model: {model_name}")
    
    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # --- Model ---
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # gradient checkpointing 需要关闭 cache
    
    # --- LoRA ---
    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # --- Data ---
    train_ds = load_dataset("json", data_files=cfg["data"]["train_file"], split="train")
    eval_ds = load_dataset("json", data_files=cfg["data"]["eval_file"], split="train")
    print(f"📊 Train: {len(train_ds)}, Eval: {len(eval_ds)}")
    
    # --- Training ---
    train_cfg = cfg["training"]
    training_args = SFTConfig(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        max_seq_length=train_cfg["max_seq_length"],
        bf16=train_cfg["bf16"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        eval_strategy=train_cfg["eval_strategy"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        report_to=train_cfg.get("report_to", "none"),
        remove_unused_columns=False,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    
    print("🏋️ Starting SFT training...")
    trainer.train()
    
    # --- Save ---
    final_dir = Path(train_cfg["output_dir"]) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"✅ Model saved to {final_dir}")
    
    # --- Quick test ---
    print("\n🧪 Quick generation test:")
    model.eval()
    test_prompt = "请解释什么是机器学习中的过拟合现象。"
    messages = [{"role": "user", "content": test_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True)
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"   Q: {test_prompt}")
    print(f"   A: {response[:200]}...")

if __name__ == "__main__":
    main()
    print("\n🎯 SFT done! Next: python scripts/3_prepare_dpo_data.py")
