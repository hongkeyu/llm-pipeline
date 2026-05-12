"""
=== Step 2: SFT — 监督微调 ===

全链路位置: Base Model → [SFT] → DPO → 量化 → 评测
目的: 在预训练模型上用指令数据微调，让模型学会"听话"（指令跟随）

核心知识点:
  1. LoRA: 冻结原始权重 W，只训练低秩矩阵 A (r×d_out) 和 B (d_in×r)
     - 前向: output = x @ W.T + x @ B @ A  (即 h = Wx + BAx)
     - 参数量: 原始 d_in×d_out → 只训练 d_in×r + r×d_out
     - 面试追问: rank 选多大？为什么 A 用高斯初始化、B 用零初始化？
  2. SFT Loss: 标准 cross-entropy，但只在 assistant 的 token 上算 loss
     - 面试追问: 为什么不在 user token 上算 loss？

你需要填写:
  - LoRALinear.forward()
  - compute_sft_loss()
"""
import json
import yaml
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# 配置加载（已写好）
# ============================================================

def load_config():
    with open("configs/base.yaml") as f:
        return yaml.safe_load(f)


# ============================================================
# LoRA 实现
# ============================================================

class LoRALinear(nn.Module):
    """
    LoRA: Low-Rank Adaptation of Large Language Models
    
    原理: 不修改预训练权重 W，而是学习一个低秩增量 ΔW = B @ A
    其中 B: (d_in, r), A: (r, d_out)，r << min(d_in, d_out)
    
    初始化:
      - A: 高斯随机初始化（N(0, σ²)）
      - B: 全零初始化
      → 训练开始时 ΔW = B @ A = 0，即从原始模型出发
    
    缩放: output = x @ W.T + (x @ B @ A) * (alpha / r)
      - alpha/r 是缩放因子，控制 LoRA 的贡献大小
      - alpha 通常设为 2*r，这样缩放因子 = 2
    """

    def __init__(self, original_layer: nn.Linear, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.original_layer = original_layer
        self.original_layer.weight.requires_grad_(False)  # 冻结原始权重
        if original_layer.bias is not None:
            original_layer.bias.requires_grad_(False)

        d_in = original_layer.in_features
        d_out = original_layer.out_features
        self.r = r
        self.scaling = alpha / r

        # LoRA 矩阵
        self.lora_B = nn.Parameter(torch.zeros(d_in, r))
        self.lora_A = nn.Parameter(torch.randn(r, d_out) * (1.0 / math.sqrt(r)))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        # TODO: Kevin实现

        LoRA 前向计算。

        输入:
            x: (batch_size, seq_len, d_in) — 输入 hidden states

        输出:
            output: (batch_size, seq_len, d_out) — 输出 hidden states

        要求:
            output = 原始线性层的输出 + LoRA 低秩增量的输出（乘以缩放因子）

        可用的成员变量:
            self.original_layer  — 原始的 nn.Linear 层（已冻结）
            self.lora_B          — (d_in, r) 低秩矩阵 B
            self.lora_A          — (r, d_out) 低秩矩阵 A
            self.scaling         — alpha / r 缩放因子
            self.dropout         — dropout 层（直接调用即可）

        Hint:
            1. 先算原始层的输出: base_output = self.original_layer(x)
            2. 对 x 做 dropout
            3. 算 LoRA 增量: x → B → A，然后乘以 scaling
            4. 两者相加

        面试追问准备:
            - 为什么 B 初始化为零？ → 保证训练开始时 ΔW=0，不破坏预训练权重
            - 为什么要缩放 alpha/r？ → 换 rank 时不用重新调 LR
            - merge 时怎么做？ → W_merged = W + B @ A * scaling，推理时无额外开销
        """
        raise NotImplementedError("请实现 LoRA forward")


def apply_lora_to_model(model, target_modules, r, alpha, dropout):
    """把模型中指定的 Linear 层替换为 LoRALinear（已写好）"""
    replaced = 0
    for name, module in model.named_modules():
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear):
                # 找到父模块并替换
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent_name, child_name = parts
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model
                    child_name = parts[0]
                
                lora_layer = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
                setattr(parent, child_name, lora_layer)
                replaced += 1

    print(f"  ✅ Replaced {replaced} layers with LoRA (r={r}, alpha={alpha})")
    
    # 统计可训练参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  📊 Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    
    return model


# ============================================================
# SFT Loss
# ============================================================

def compute_sft_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    # TODO: Kevin实现

    计算 SFT 的 cross-entropy loss，只在 assistant tokens 上算。

    输入:
        logits: (batch_size, seq_len, vocab_size) — 模型输出的 logits
        labels: (batch_size, seq_len) — 目标 token ids
                其中 user/system 的 token 位置被设为 ignore_index (-100)
                只有 assistant 的 token 位置是真实的 token id

    输出:
        loss: scalar tensor — 平均 cross-entropy loss

    Hint:
        1. logits 和 labels 有一个 offset 关系：
           logits[t] 预测的是 labels[t+1]（自回归！）
           所以需要: logits = logits[:, :-1, :] 和 labels = labels[:, 1:]
        2. 用 F.cross_entropy，注意设置 ignore_index=-100
        3. 记得 reshape — cross_entropy 期望 (N, C) 和 (N,)

    面试追问准备:
        - 为什么只在 assistant tokens 上算 loss？
          → user tokens 是输入条件，不是要学的内容；在 user 上算 loss 会浪费梯度
        - 这和 CLM 预训练有什么区别？
          → 预训练对所有 token 算 loss，SFT 只对回答部分算
    """
    raise NotImplementedError("请实现 SFT loss")


# ============================================================
# 数据集（已写好）
# ============================================================

class SFTDataset(Dataset):
    """把 messages 转成 input_ids + labels（user 部分 mask 为 -100）"""

    def __init__(self, data_path: str, tokenizer, max_seq_len: int):
        self.samples = []
        with open(data_path) as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        messages = self.samples[idx]["messages"]

        # 用 chat template 编码完整对话
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        full_ids = self.tokenizer(full_text, truncation=True, max_length=self.max_seq_len, return_tensors="pt")
        input_ids = full_ids["input_ids"].squeeze(0)

        # 构造 labels: user 部分 = -100, assistant 部分 = token id
        # 策略: 找到 assistant 回答的起始位置，之前的全部 mask
        labels = input_ids.clone()

        # 只编码 user 部分，找到分界点
        user_text = self.tokenizer.apply_chat_template(
            [messages[0]], tokenize=False, add_generation_prompt=True
        )
        user_ids = self.tokenizer(user_text, truncation=True, max_length=self.max_seq_len, return_tensors="pt")
        user_len = user_ids["input_ids"].shape[1]

        # user 部分的 labels 设为 -100（不算 loss）
        labels[:user_len] = -100

        return {"input_ids": input_ids, "labels": labels}


def collate_fn(batch, pad_token_id):
    """动态 padding（已写好）"""
    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].shape[0]
        input_ids[i, :seq_len] = item["input_ids"]
        labels[i, :seq_len] = item["labels"]
        attention_mask[i, :seq_len] = 1

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


# ============================================================
# 训练循环（已写好，调用你实现的 forward 和 loss）
# ============================================================

def train():
    cfg = load_config()
    sft_cfg = cfg["sft"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型和 tokenizer
    print(f"🚀 Loading {cfg['model']['name']}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # 应用 LoRA
    print("🔧 Applying LoRA...")
    model = apply_lora_to_model(
        model,
        target_modules=sft_cfg["target_modules"],
        r=sft_cfg["lora_r"],
        alpha=sft_cfg["lora_alpha"],
        dropout=sft_cfg["lora_dropout"],
    )

    # 数据
    train_ds = SFTDataset("data/sft/train.jsonl", tokenizer, sft_cfg["max_seq_len"])
    val_ds = SFTDataset("data/sft/val.jsonl", tokenizer, sft_cfg["max_seq_len"])

    from functools import partial
    collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=sft_cfg["batch_size"], shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=sft_cfg["batch_size"], collate_fn=collate)

    # 优化器（只优化 LoRA 参数）
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=sft_cfg["lr"], weight_decay=0.01)

    # Cosine LR scheduler
    total_steps = len(train_loader) * sft_cfg["epochs"] // sft_cfg["grad_accum"]
    warmup_steps = int(total_steps * 0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # 训练
    output_dir = Path(sft_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🏋️ Training for {sft_cfg['epochs']} epochs, {total_steps} steps...")
    global_step = 0
    for epoch in range(sft_cfg["epochs"]):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward — 这里会调用你实现的 LoRALinear.forward()
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

            # Loss — 调用你实现的 compute_sft_loss()
            loss = compute_sft_loss(outputs.logits, batch["labels"])
            loss = loss / sft_cfg["grad_accum"]
            loss.backward()

            if (step + 1) % sft_cfg["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    print(f"  [Epoch {epoch+1}] Step {global_step}/{total_steps} | Loss: {loss.item() * sft_cfg['grad_accum']:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

            epoch_loss += loss.item() * sft_cfg["grad_accum"]

        avg_loss = epoch_loss / len(train_loader)
        print(f"  📊 Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = compute_sft_loss(outputs.logits, batch["labels"])
                val_loss += loss.item()
        print(f"  📊 Epoch {epoch+1} val loss: {val_loss / len(val_loader):.4f}")

    # 保存
    save_dir = output_dir / "final"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存 LoRA 权重
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[f"{name}.lora_A"] = module.lora_A.data.cpu()
            lora_state[f"{name}.lora_B"] = module.lora_B.data.cpu()
    torch.save(lora_state, save_dir / "lora_weights.pt")
    tokenizer.save_pretrained(save_dir)
    print(f"✅ LoRA weights saved to {save_dir}")


if __name__ == "__main__":
    train()
    print("\n🎯 SFT done! Next: python scripts/3_dpo_train.py")
