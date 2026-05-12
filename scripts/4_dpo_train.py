"""
Step 4: DPO Training
偏好优化 — 在 SFT 模型基础上做对齐
"""
import yaml
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from trl import DPOTrainer, DPOConfig

def load_config():
    with open("configs/dpo_config.yaml") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    model_path = cfg["model"]["name"]
    
    print(f"🚀 Loading SFT model: {model_path}")
    
    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # --- Model ---
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
        device_map="auto",
        trust_remote_code=True,
    )
    
    # --- Reference Model (frozen copy) ---
    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["ref_model"],
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
        device_map="auto",
        trust_remote_code=True,
    )
    
    # --- LoRA ---
    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=TaskType.CAUSAL_LM,
    )
    
    # --- Data ---
    data_cfg = cfg["data"]
    train_ds = load_dataset("json", data_files=data_cfg["train_file"], split="train")
    eval_ds = load_dataset("json", data_files=data_cfg["eval_file"], split="train")
    print(f"📊 Train: {len(train_ds)} pairs, Eval: {len(eval_ds)} pairs")
    
    # --- DPO Training ---
    train_cfg = cfg["training"]
    dpo_cfg = cfg["dpo"]
    
    training_args = DPOConfig(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        max_length=train_cfg["max_length"],
        max_prompt_length=train_cfg["max_prompt_length"],
        bf16=train_cfg["bf16"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        report_to=train_cfg.get("report_to", "none"),
        beta=dpo_cfg["beta"],
        loss_type=dpo_cfg["loss_type"],
        remove_unused_columns=False,
    )
    
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    print(f"🏋️ Starting DPO training (β={dpo_cfg['beta']})...")
    trainer.train()
    
    # --- Save ---
    final_dir = Path(train_cfg["output_dir"]) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"✅ DPO model saved to {final_dir}")
    
    # --- Log metrics ---
    metrics = trainer.state.log_history
    reward_accuracies = [m.get("eval_rewards/accuracies", None) for m in metrics if m.get("eval_rewards/accuracies")]
    if reward_accuracies:
        print(f"📈 Final reward accuracy: {reward_accuracies[-1]:.4f}")

if __name__ == "__main__":
    main()
    print("\n🎯 DPO done! Next: python scripts/5_quantize.py")
