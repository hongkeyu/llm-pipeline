"""
=== Step 4: Quantization — 模型量化 ===

全链路位置: Base Model → SFT → DPO → [量化] → 评测
目的: 压缩模型大小、加速推理、降低显存占用

核心知识点:
  PTQ (Post-Training Quantization): 训练后量化，不需要重新训练
  
  量化 = 把 FP16/FP32 的浮点权重映射到 INT8/INT4 的整数空间
  
  两种量化方式:
    1. 对称量化: x_q = round(x / scale)
       - scale = max(|x|) / (2^(bits-1) - 1)
       - 反量化: x_dequant = x_q * scale
       - 简单，但如果分布不对称会浪费一半范围
    
    2. 非对称量化: x_q = round(x / scale) + zero_point
       - scale = (max(x) - min(x)) / (2^bits - 1)
       - zero_point = round(-min(x) / scale)
       - 反量化: x_dequant = (x_q - zero_point) * scale
       - 能更好地利用量化范围
  
  Group Quantization:
    - 不是整个 tensor 共享一组 (scale, zero_point)
    - 而是每 group_size 个元素共享一组
    - 精度更高，但多存了 scale/zero_point 的开销

你需要填写:
  - compute_quantization_params(): 计算 scale 和 zero_point
  - fake_quantize(): 量化→反量化模拟
  - quantize_model_weights(): 逐层分组量化
"""
import yaml
import json
import copy
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config():
    with open("configs/base.yaml") as f:
        return yaml.safe_load(f)


# ============================================================
# 量化核心函数
# ============================================================

def compute_quantization_params(
    tensor: torch.Tensor,
    bits: int,
    symmetric: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    # TODO: Kevin实现

    计算量化参数 scale 和 zero_point。

    输入:
        tensor: (N,) — 要量化的一维浮点 tensor
        bits: int — 目标位数（4 或 8）
        symmetric: bool — 对称量化 or 非对称量化

    输出:
        scale: scalar tensor — 量化缩放因子
        zero_point: scalar tensor — 零点偏移（对称量化时为 0）

    === 对称量化 (symmetric=True) ===
    
        量化范围: [-2^(bits-1), 2^(bits-1) - 1]
        例如 INT8: [-128, 127]
        
        scale = max(|tensor|) / (2^(bits-1) - 1)
        zero_point = 0
        
        量化: x_q = clamp(round(x / scale), -2^(bits-1), 2^(bits-1)-1)
        反量化: x ≈ x_q * scale

    === 非对称量化 (symmetric=False) ===
    
        量化范围: [0, 2^bits - 1]
        例如 UINT8: [0, 255]
        
        scale = (max(tensor) - min(tensor)) / (2^bits - 1)
        zero_point = round(-min(tensor) / scale)
        
        量化: x_q = clamp(round(x / scale + zero_point), 0, 2^bits - 1)
        反量化: x ≈ (x_q - zero_point) * scale

    Hint:
        1. 先算 qmin, qmax（量化范围的最小最大值）
        2. 对称: scale = max(|tensor|) / qmax, zero_point = 0
        3. 非对称: scale = (tensor.max() - tensor.min()) / (qmax - qmin)
                   zero_point = round(-tensor.min() / scale)
        4. scale 加一个 eps (1e-8) 防止除零
        5. zero_point clamp 到 [qmin, qmax]

    面试追问:
        - 对称 vs 非对称各自什么场景好？
          → 权重通常近似对称分布用对称；激活值往往有偏移用非对称
        - 为什么 GPTQ 用非对称？
          → 逐列量化，每列分布可能有偏移
        - 为什么 INT4 比 INT8 掉点更多？
          → 量化范围从 256 个值压到 16 个，每个值代表更大的范围
    """
    raise NotImplementedError("请实现 compute_quantization_params")


def fake_quantize(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    bits: int,
    symmetric: bool = True,
) -> torch.Tensor:
    """
    # TODO: Kevin实现

    Fake quantization: 量化 → 反量化（模拟量化误差）。

    输入:
        tensor: (...) — 原始浮点 tensor（任意形状）
        scale: scalar — 缩放因子
        zero_point: scalar — 零点
        bits: int — 位数
        symmetric: bool — 对称 or 非对称

    输出:
        dequantized: (...) — 与输入同形状的 tensor，模拟量化后的值

    过程:
        1. 量化:   x_q = clamp(round(x / scale + zero_point), qmin, qmax)
        2. 反量化: x_dequant = (x_q - zero_point) * scale

    这就是 PTQ 的核心 — 模拟一下量化会带来多大误差。
    实际部署时 x_q 就是存储的整数值，推理时用整数运算。

    Hint:
        - 对称: qmin = -2^(bits-1), qmax = 2^(bits-1) - 1, zero_point = 0
        - 非对称: qmin = 0, qmax = 2^bits - 1
        - torch.round() 做舍入
        - torch.clamp() 做截断

    面试追问:
        - fake quantize 和 real quantize 有什么区别？
          → fake 保持 FP32/FP16 数据类型，只模拟精度损失；real 转成 INT 存储节省空间
        - QAT 和 PTQ 的区别？
          → QAT 在训练中插入 fake quantize，让模型学会适应量化误差；PTQ 不重新训练
        - 为什么 STE (Straight-Through Estimator) 重要？
          → round() 梯度为 0，QAT 用 STE 让梯度穿过 round 操作
    """
    raise NotImplementedError("请实现 fake_quantize")


def quantize_model_weights(
    model: nn.Module,
    bits: int,
    group_size: int = 128,
    symmetric: bool = True,
) -> tuple[nn.Module, dict]:
    """
    # TODO: Kevin实现

    对模型所有 Linear 层的权重做分组量化 (group quantization)。

    输入:
        model: nn.Module — 要量化的模型
        bits: int — 目标位数
        group_size: int — 每组多少个元素共享一组 scale/zero_point
        symmetric: bool — 量化方式

    输出:
        model: nn.Module — 权重被替换为 fake-quantized 版本的模型
        stats: dict — 量化统计信息
            {
                "num_layers_quantized": int,
                "avg_error": float,  # 平均量化误差 (MSE)
                "max_error": float,  # 最大量化误差
            }

    Group Quantization 过程:
        对每个 Linear 层的 weight (out_features, in_features):
        1. 把 weight 按行 reshape 成 (-1, group_size)
           例如 (768, 768) with group_size=128 → (768*768/128, 128) = (4608, 128)
        2. 对每一组（每一行）独立计算 scale 和 zero_point
        3. 对每一组做 fake_quantize
        4. reshape 回原来的形状
        5. 替换 layer.weight.data

    Hint:
        1. 遍历 model.named_modules()，找到所有 nn.Linear
        2. weight = layer.weight.data
        3. 如果 in_features 不能被 group_size 整除，最后一组特殊处理（或 pad）
        4. 记录每层的量化误差: error = (original - quantized).pow(2).mean()
        5. 调用你已经实现的 compute_quantization_params 和 fake_quantize

    面试追问:
        - group_size 怎么选？
          → 越小精度越高但存储 scale/zp 的开销越大；128 是常见平衡点
        - GPTQ 和简单 PTQ 有什么区别？
          → GPTQ 用 Hessian 信息逐列量化，误差补偿到后续列（OBQ 的近似）
        - AWQ 的思路？
          → 找到对激活值影响大的"显著权重"，保护它们不被过度量化
    """
    raise NotImplementedError("请实现 quantize_model_weights")


# ============================================================
# 量化评估（已写好）
# ============================================================

def evaluate_quantization_error(original_model, quantized_model, tokenizer, test_texts):
    """对比量化前后的输出差异"""
    device = next(original_model.parameters()).device
    results = []

    for text in test_texts[:10]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)

        with torch.no_grad():
            orig_logits = original_model(**inputs).logits
            quant_logits = quantized_model(**inputs).logits

        # KL divergence
        orig_probs = torch.softmax(orig_logits, dim=-1)
        quant_log_probs = torch.log_softmax(quant_logits, dim=-1)
        kl_div = torch.nn.functional.kl_div(quant_log_probs, orig_probs, reduction="batchmean").item()

        # 输出 token 一致性
        orig_tokens = orig_logits.argmax(dim=-1)
        quant_tokens = quant_logits.argmax(dim=-1)
        agreement = (orig_tokens == quant_tokens).float().mean().item()

        results.append({"kl_div": kl_div, "token_agreement": agreement})

    avg_kl = sum(r["kl_div"] for r in results) / len(results)
    avg_agree = sum(r["token_agreement"] for r in results) / len(results)
    return {"avg_kl_div": avg_kl, "avg_token_agreement": avg_agree}


# ============================================================
# 主流程（已写好）
# ============================================================

def main():
    cfg = load_config()
    quant_cfg = cfg["quantization"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = cfg["model"]["name"]
    print(f"🚀 Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map=device, trust_remote_code=True,
    )
    original_model.eval()

    # 测试文本
    test_texts = [
        "人工智能的未来发展方向包括",
        "量化是一种模型压缩技术",
        "Transformer架构的核心是自注意力机制",
        "LoRA通过低秩分解减少微调参数量",
        "大语言模型的推理优化方法有很多",
    ]

    all_results = {}

    for bits in quant_cfg["bits"]:
        print(f"\n{'='*60}")
        print(f"INT{bits} Quantization (group_size={quant_cfg['group_size']})")
        print(f"{'='*60}")

        # 复制模型
        quant_model = copy.deepcopy(original_model)

        # 调用你实现的量化函数
        quant_model, stats = quantize_model_weights(
            quant_model,
            bits=bits,
            group_size=quant_cfg["group_size"],
        )

        print(f"  📊 Layers quantized: {stats['num_layers_quantized']}")
        print(f"  📊 Avg quantization error (MSE): {stats['avg_error']:.6f}")
        print(f"  📊 Max quantization error (MSE): {stats['max_error']:.6f}")

        # 评估
        eval_results = evaluate_quantization_error(original_model, quant_model, tokenizer, test_texts)
        print(f"  📊 Avg KL divergence: {eval_results['avg_kl_div']:.4f}")
        print(f"  📊 Token agreement: {eval_results['avg_token_agreement']:.2%}")

        # 模型大小估算
        orig_size = sum(p.numel() * p.element_size() for p in original_model.parameters()) / 1024**2
        quant_size = orig_size * (bits / 16)  # 粗略估算
        print(f"  📊 Size: {orig_size:.1f} MB (FP16) → ~{quant_size:.1f} MB (INT{bits})")

        all_results[f"INT{bits}"] = {**stats, **eval_results, "estimated_size_mb": quant_size}

        del quant_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 保存结果
    output_dir = Path(quant_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # 打印对比
    print(f"\n{'='*60}")
    print("📊 Quantization Comparison")
    print(f"{'='*60}")
    print(f"{'Method':<10} {'MSE':>10} {'KL Div':>10} {'Token Agr':>10} {'Size (MB)':>10}")
    print("-" * 55)
    for method, r in all_results.items():
        print(f"{method:<10} {r['avg_error']:>10.6f} {r['avg_kl_div']:>10.4f} "
              f"{r['avg_token_agreement']:>10.2%} {r['estimated_size_mb']:>10.1f}")


if __name__ == "__main__":
    main()
    print("\n🎯 Quantization done! Next: python scripts/5_evaluate.py")
