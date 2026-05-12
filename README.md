# LLM Full Pipeline — 填空式实操项目

> **目的**：通过手写核心逻辑理解 LLM 后训练全链路，面试被追问时能说出实现细节。

## Pipeline

```
Qwen3-0.6B (Base)
    │
    ├─ [Step 1] SFT ── 手写 LoRA forward + training loop
    │
    ├─ [Step 2] DPO ── 手写 DPO loss + preference optimization
    │
    ├─ [Step 3] Quantization ── 手写 PTQ scale/zero_point + fake quantize
    │
    └─ [Step 4] Evaluation ── 手写 perplexity + generation quality metrics
```

## 规则

- **`# TODO: Kevin实现`** = 你需要写的核心代码，有输入输出说明和 hint
- **其他代码** = 已写好的脚手架（数据加载、config、模型保存、日志等）
- 每个 script 开头有注释解释这一步在全链路中的位置

## 运行顺序

```bash
python scripts/1_prepare_data.py          # 数据准备（已写好）
python scripts/2_sft_train.py             # SFT 训练（需要填 LoRA forward）
python scripts/3_dpo_train.py             # DPO 训练（需要填 DPO loss）
python scripts/4_quantize.py              # 量化（需要填 PTQ 参数计算 + fake quantize）
python scripts/5_evaluate.py              # 评测（需要填 metrics 计算）
```

## TODO 清单

| Script | TODO | 核心知识点 |
|--------|------|-----------|
| `2_sft_train.py` | `LoRALinear.forward()` | LoRA 低秩分解前向计算 |
| `2_sft_train.py` | `compute_sft_loss()` | SFT loss mask（只算 assistant tokens） |
| `3_dpo_train.py` | `get_per_token_logps()` | 从 logits 提取指定 token 的 log probability |
| `3_dpo_train.py` | `compute_dpo_loss()` | DPO 公式实现 |
| `4_quantize.py` | `compute_quantization_params()` | 对称/非对称量化参数计算 |
| `4_quantize.py` | `fake_quantize()` | 量化→反量化模拟（PTQ 核心） |
| `4_quantize.py` | `quantize_model_weights()` | 逐层量化 + group quantization |
| `5_evaluate.py` | `compute_perplexity()` | 从 logits 算 PPL |
| `5_evaluate.py` | `compute_bleu()` | n-gram precision + brevity penalty |
| `5_evaluate.py` | `compute_diversity()` | distinct n-gram 多样性指标 |
| `5_evaluate.py` | `compute_win_rate()` | DPO 前后 reward 对比 |

## 面试叙事

**L1 (60s)**："我从零手写了 LoRA、DPO loss、PTQ 量化的核心逻辑，在 Qwen3-0.6B 上跑通了 SFT→DPO→量化→评测全流程。"

**L2 (追问)**：每个 TODO 的实现就是你的答案。

**L3 (深度)**：消融实验 — LoRA rank / DPO β / 量化 bit 数的 trade-off 曲线。
