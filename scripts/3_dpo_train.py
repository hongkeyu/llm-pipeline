"""
=== Step 3: DPO — Direct Preference Optimization ===

全链路位置: Base Model → SFT → [DPO] → 量化 → 评测
目的: 让模型输出更符合人类偏好（对齐）

背景:
  RLHF 需要 4 个模型（policy, ref, reward, critic），工程复杂度极高。
  DPO 把 reward model 隐式地融进了 policy 优化中，只需要 2 个模型（policy, ref）。

核心公式:
  L_DPO = -E[log σ(β · (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]

  其中:
    π     = 当前策略（正在训练的模型）
    π_ref = 参考策略（SFT 后冻结的模型）
    y_w   = chosen（人类偏好的回答）
    y_l   = rejected（人类不偏好的回答）
    β     = KL penalty 系数

  直觉: 拉大 chosen 和 rejected 的对数概率差，同时不偏离 ref 太远

你需要填写:
  - get_per_token_logps(): 从 logits 提取 token-level log probabilities
  - compute_dpo_loss(): DPO loss 完整实现
"""
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config():
    with open("configs/base.yaml") as f:
        return yaml.safe_load(f)


# ============================================================
# DPO 核心函数
# ============================================================

def get_per_token_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    # TODO: Kevin实现

    从模型输出的 logits 中，提取每个位置对应 label token 的 log probability。

    输入:
        logits: (batch_size, seq_len, vocab_size) — 模型的原始输出
        labels: (batch_size, seq_len) — 目标 token ids

    输出:
        log_probs: (batch_size, seq_len - 1) — 每个位置的 log P(label_t | x_{<t})

    注意自回归 offset:
        logits[t] 预测的是 token[t+1]
        所以: logits[:, :-1, :] 对应 labels[:, 1:]

    Hint:
        1. 先做 offset: logits = logits[:, :-1, :], labels = labels[:, 1:]
        2. 对 logits 做 log_softmax（在 vocab 维度）
        3. 用 torch.gather 提取 labels 对应位置的 log prob
        4. squeeze 掉最后一维

    面试追问:
        - 为什么用 log_softmax 而不是先 softmax 再 log？
          → 数值稳定性：log_softmax 内部用 log-sum-exp trick 避免上溢/下溢
        - 为什么是 per-token 而不是直接算 sequence-level？
          → DPO 需要对 response 部分的 token 求和，prompt 部分要 mask 掉
    """
    raise NotImplementedError("请实现 get_per_token_logps")


def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    # TODO: Kevin实现

    计算 DPO loss。

    输入:
        policy_chosen_logps:   (batch_size,) — π(y_w | x) 的 log prob（response 部分求和）
        policy_rejected_logps: (batch_size,) — π(y_l | x) 的 log prob（response 部分求和）
        ref_chosen_logps:      (batch_size,) — π_ref(y_w | x) 的 log prob
        ref_rejected_logps:    (batch_size,) — π_ref(y_l | x) 的 log prob
        beta: float — KL penalty 系数

    输出:
        loss:    scalar — DPO loss（batch 平均）
        chosen_rewards:  (batch_size,) — β * (log π/π_ref) for chosen
        rejected_rewards: (batch_size,) — β * (log π/π_ref) for rejected

    DPO 公式:
        log_ratio_chosen  = policy_chosen_logps - ref_chosen_logps     # log(π/π_ref) for y_w
        log_ratio_rejected = policy_rejected_logps - ref_rejected_logps # log(π/π_ref) for y_l
        logits = β * (log_ratio_chosen - log_ratio_rejected)
        loss = -log(σ(logits))   即 F.binary_cross_entropy_with_logits(logits, ones)

    Hint:
        1. 算两个 log ratio
        2. 相减，乘以 β
        3. 过 logsigmoid（或用 -F.logsigmoid(logits)）
        4. 取 batch 平均
        5. rewards = β * log_ratio（用于监控训练，不参与 loss 计算）

    面试追问:
        - β 太大会怎样？ → 模型更不敢偏离 ref，保守但安全
        - β 太小会怎样？ → 模型偏离 ref 太远，可能 reward hacking
        - DPO 和 PPO 的区别？ → DPO 是离线的（数据固定），PPO 在线采样
        - DPO 的缺陷？ → 受限于离线数据的质量和多样性
    """
    raise NotImplementedError("请实现 DPO loss")


# ============================================================
# 数据集（已写好）
# ============================================================

class DPODataset(Dataset):
    """加载偏好对数据，编码 prompt + chosen 和 prompt + rejected"""

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
        item = self.samples[idx]

        # 编码 prompt + chosen
        chosen_messages = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["chosen"]},
        ]
        chosen_text = self.tokenizer.apply_chat_template(chosen_messages, tokenize=False, add_generation_prompt=False)
        chosen_enc = self.tokenizer(chosen_text, truncation=True, max_length=self.max_seq_len, return_tensors="pt")

        # 编码 prompt + rejected
        rejected_messages = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["rejected"]},
        ]
        rejected_text = self.tokenizer.apply_chat_template(rejected_messages, tokenize=False, add_generation_prompt=False)
        rejected_enc = self.tokenizer(rejected_text, truncation=True, max_length=self.max_seq_len, return_tensors="pt")

        # 找到 prompt 长度（用于 mask）
        prompt_messages = [{"role": "user", "content": item["prompt"]}]
        prompt_text = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        prompt_enc = self.tokenizer(prompt_text, truncation=True, max_length=self.max_seq_len, return_tensors="pt")
        prompt_len = prompt_enc["input_ids"].shape[1]

        return {
            "chosen_ids": chosen_enc["input_ids"].squeeze(0),
            "rejected_ids": rejected_enc["input_ids"].squeeze(0),
            "prompt_len": prompt_len,
        }


def dpo_collate_fn(batch, pad_token_id):
    """分别 pad chosen 和 rejected（已写好）"""
    chosen_max = max(item["chosen_ids"].shape[0] for item in batch)
    rejected_max = max(item["rejected_ids"].shape[0] for item in batch)

    chosen_ids = torch.full((len(batch), chosen_max), pad_token_id, dtype=torch.long)
    rejected_ids = torch.full((len(batch), rejected_max), pad_token_id, dtype=torch.long)
    chosen_mask = torch.zeros(len(batch), chosen_max, dtype=torch.long)
    rejected_mask = torch.zeros(len(batch), rejected_max, dtype=torch.long)
    prompt_lens = torch.zeros(len(batch), dtype=torch.long)

    for i, item in enumerate(batch):
        c_len = item["chosen_ids"].shape[0]
        r_len = item["rejected_ids"].shape[0]
        chosen_ids[i, :c_len] = item["chosen_ids"]
        rejected_ids[i, :r_len] = item["rejected_ids"]
        chosen_mask[i, :c_len] = 1
        rejected_mask[i, :r_len] = 1
        prompt_lens[i] = item["prompt_len"]

    return {
        "chosen_ids": chosen_ids,
        "rejected_ids": rejected_ids,
        "chosen_mask": chosen_mask,
        "rejected_mask": rejected_mask,
        "prompt_lens": prompt_lens,
    }


# ============================================================
# 计算 sequence-level log probs（已写好，调用你实现的 get_per_token_logps）
# ============================================================

def get_sequence_logps(model, input_ids, attention_mask, prompt_lens):
    """
    计算 response 部分的 log probability 总和。
    prompt 部分被 mask 掉，只对 response tokens 求和。
    """
    with torch.no_grad() if not model.training else torch.enable_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # 调用你实现的函数
    per_token_logps = get_per_token_logps(outputs.logits, input_ids)  # (B, seq_len-1)

    # 构建 response mask: prompt 部分为 0，response 部分为 1
    # 注意 per_token_logps 比 input_ids 短 1（因为 offset）
    response_mask = torch.zeros_like(per_token_logps)
    for i in range(len(prompt_lens)):
        # prompt_lens[i] 是 prompt 的 token 数
        # offset 后，response 从 prompt_lens[i]-1 开始
        start = max(prompt_lens[i] - 1, 0)
        end = attention_mask[i].sum() - 1  # 去掉 padding
        response_mask[i, start:end] = 1

    # 只对 response 部分求和
    sequence_logps = (per_token_logps * response_mask).sum(dim=1)
    return sequence_logps


# ============================================================
# 训练循环（已写好，调用你实现的 DPO loss）
# ============================================================

def train():
    cfg = load_config()
    dpo_cfg = cfg["dpo"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 SFT 后的模型作为初始策略
    sft_path = Path(cfg["sft"]["output_dir"]) / "final"
    print(f"🚀 Loading SFT model from {sft_path}...")

    tokenizer = AutoTokenizer.from_pretrained(sft_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Policy model (会被训练)
    policy_model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"], torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    # 加载 LoRA weights
    lora_state = torch.load(sft_path / "lora_weights.pt", map_location=device)
    # TODO: 应用 LoRA weights 到 policy_model（可以复用 Step 2 的 apply_lora_to_model）

    # Reference model (冻结，不训练)
    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"], torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # 数据
    from functools import partial
    collate = partial(dpo_collate_fn, pad_token_id=tokenizer.pad_token_id)
    train_ds = DPODataset("data/dpo/train.jsonl", tokenizer, dpo_cfg["max_seq_len"])
    train_loader = DataLoader(train_ds, batch_size=dpo_cfg["batch_size"], shuffle=True, collate_fn=collate)

    # 优化器
    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=dpo_cfg["lr"])

    total_steps = len(train_loader) * dpo_cfg["epochs"] // dpo_cfg["grad_accum"]
    print(f"🏋️ DPO training: {total_steps} steps, β={dpo_cfg['beta']}...")

    global_step = 0
    for epoch in range(dpo_cfg["epochs"]):
        policy_model.train()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Policy log probs
            policy_chosen_logps = get_sequence_logps(
                policy_model, batch["chosen_ids"], batch["chosen_mask"], batch["prompt_lens"]
            )
            policy_rejected_logps = get_sequence_logps(
                policy_model, batch["rejected_ids"], batch["rejected_mask"], batch["prompt_lens"]
            )

            # Reference log probs (no grad)
            with torch.no_grad():
                ref_chosen_logps = get_sequence_logps(
                    ref_model, batch["chosen_ids"], batch["chosen_mask"], batch["prompt_lens"]
                )
                ref_rejected_logps = get_sequence_logps(
                    ref_model, batch["rejected_ids"], batch["rejected_mask"], batch["prompt_lens"]
                )

            # 调用你实现的 DPO loss
            loss, chosen_rewards, rejected_rewards = compute_dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps,
                beta=dpo_cfg["beta"],
            )

            loss = loss / dpo_cfg["grad_accum"]
            loss.backward()

            if (step + 1) % dpo_cfg["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                reward_acc = (chosen_rewards > rejected_rewards).float().mean().item()
                if global_step % 20 == 0:
                    print(
                        f"  [Step {global_step}/{total_steps}] "
                        f"Loss: {loss.item() * dpo_cfg['grad_accum']:.4f} | "
                        f"Reward Acc: {reward_acc:.2%} | "
                        f"Chosen R: {chosen_rewards.mean().item():.3f} | "
                        f"Rejected R: {rejected_rewards.mean().item():.3f}"
                    )

    # 保存
    save_dir = Path(dpo_cfg["output_dir"]) / "final"
    save_dir.mkdir(parents=True, exist_ok=True)
    policy_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"✅ DPO model saved to {save_dir}")


if __name__ == "__main__":
    train()
    print("\n🎯 DPO done! Next: python scripts/4_quantize.py")
