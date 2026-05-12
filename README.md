# LLM Full Pipeline: SFT → DPO → Quantization → Deployment → Evaluation

> 面试展示项目：大模型后训练全链路实操能力

## 项目定位

面试时一句话介绍：**"我从零搭建了一个覆盖 SFT、DPO 对齐、量化压缩、推理部署、评测的完整 pipeline，在 Qwen3-0.6B 上跑通了全流程，每个环节都有量化对比和消融实验。"**

## Pipeline Overview

```
Qwen3-0.6B (Base)
    │
    ▼
[Step 1] SFT — LoRA 指令微调
    │   数据: Alpaca-zh + 自定义指令集
    │   工具: transformers + peft + trl
    │   关键指标: loss curve, 指令跟随率
    │
    ▼
[Step 2] DPO — 偏好对齐
    │   数据: 构建 chosen/rejected pairs
    │   工具: trl DPOTrainer
    │   关键指标: reward accuracy, win rate vs SFT
    │
    ▼
[Step 3] Quantization — 量化压缩
    │   方案: GPTQ (INT4/INT8), AWQ, bitsandbytes
    │   对比: 精度 vs 速度 vs 显存
    │   关键指标: perplexity degradation, throughput
    │
    ▼
[Step 4] Deployment — 推理服务
    │   引擎: vLLM (GPU) / llama.cpp (CPU/边缘)
    │   关键指标: TTFT, TPOT, throughput (tokens/s)
    │
    ▼
[Step 5] Evaluation — 全流程评测
        基础能力: MMLU, C-Eval, HumanEval
        对齐质量: MT-Bench, AlpacaEval
        对比: Base vs SFT vs DPO vs Quantized
```

## 面试叙事线

### L1 (60s 概述)
"我做了一个 LLM 全链路项目，从 SFT 微调到 DPO 对齐到量化部署，在 Qwen3-0.6B 上跑通全流程。SFT 用 LoRA 在 5K 条中文指令数据上微调，DPO 用我构建的偏好对做对齐训练，然后对比了 GPTQ INT4/INT8 的精度-速度 trade-off，最后用 vLLM 部署测了 TTFT 和吞吐量。"

### L2 (3min 技术深度)
每个环节的技术选型理由、遇到的问题、数据处理细节。

### L3 (追问级)
- LoRA rank 选择的消融实验
- DPO 的 β 参数敏感性
- GPTQ vs AWQ 的量化策略差异
- KV Cache 优化对 TTFT 的影响
- Continuous Batching 对吞吐的提升

## Directory Structure

```
llm-pipeline/
├── README.md
├── configs/
│   ├── sft_config.yaml       # SFT 训练配置
│   ├── dpo_config.yaml       # DPO 训练配置
│   └── quant_config.yaml     # 量化配置
├── scripts/
│   ├── 1_prepare_sft_data.py     # SFT 数据准备
│   ├── 2_sft_train.py            # SFT 训练
│   ├── 3_prepare_dpo_data.py     # DPO 偏好对构建
│   ├── 4_dpo_train.py            # DPO 训练
│   ├── 5_quantize.py             # 量化
│   ├── 6_deploy_vllm.py          # vLLM 部署
│   ├── 7_deploy_llamacpp.py      # llama.cpp 部署
│   ├── 8_evaluate.py             # 评测
│   └── 9_benchmark.py            # 推理性能基准测试
├── data/
│   ├── sft/                  # SFT 训练数据
│   ├── dpo/                  # DPO 偏好对数据
│   └── eval/                 # 评测数据
├── notebooks/
│   └── analysis.ipynb        # 结果分析可视化
├── outputs/
│   ├── sft/                  # SFT checkpoints
│   ├── dpo/                  # DPO checkpoints
│   ├── quantized/            # 量化模型
│   └── eval/                 # 评测结果
└── requirements.txt
```

## Quick Start

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据
python scripts/1_prepare_sft_data.py
python scripts/3_prepare_dpo_data.py

# 3. SFT 训练 (单卡 A100/4090, ~1h)
python scripts/2_sft_train.py

# 4. DPO 训练 (~30min)
python scripts/4_dpo_train.py

# 5. 量化
python scripts/5_quantize.py

# 6. 部署 & 评测
python scripts/6_deploy_vllm.py
python scripts/8_evaluate.py
python scripts/9_benchmark.py
```

## Hardware Requirements

- **训练**: 单张 A100 40GB 或 RTX 4090 24GB（Qwen3-0.6B + LoRA 足够）
- **量化**: 同上
- **推理**: CPU 也可跑（llama.cpp），GPU 推理用 vLLM
- **Jetson Orin Nano 8GB**: 可跑 INT4 量化后的推理（llama.cpp），不建议训练

## 技术选型理由 (面试用)

| 决策 | 选择 | 理由 |
|------|------|------|
| Base Model | Qwen3-0.6B | 单卡可跑全流程，Qwen3 架构新（GQA+SwiGLU+RMSNorm），中文能力强 |
| SFT 方法 | LoRA (r=16) | 显存友好，效果接近全参微调，面试可以讲 rank 选择的 trade-off |
| 对齐方法 | DPO | Kevin 有 DPO 研究经验（M2H MeLLo），面试可以深入讲公式推导和 β 敏感性 |
| 量化 | GPTQ + AWQ 对比 | 两种主流 PTQ 方案，面试可以讲量化策略差异 |
| 推理引擎 | vLLM + llama.cpp | vLLM 展示 PagedAttention/Continuous Batching，llama.cpp 展示端侧部署 |
