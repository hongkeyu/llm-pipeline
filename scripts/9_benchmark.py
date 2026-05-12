"""
Step 9: Inference Benchmark
对比不同量化方案的推理性能 (TTFT, TPOT, Throughput)
"""
import json
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS_DIR = Path("outputs/eval")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def benchmark_hf_model(model_path: str, label: str, quantization: str = None):
    """
    HuggingFace 推理性能测试
    
    面试要点:
    - TTFT (Time To First Token): prefill 阶段，受输入长度影响
    - TPOT (Time Per Output Token): decode 阶段，受显存带宽影响
    - Throughput: 每秒生成 token 数
    - 量化后 TPOT 通常更快（权重更小→显存带宽更高）
    """
    print(f"\n📊 Benchmarking [{label}]: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    load_kwargs = {"device_map": "auto", "trust_remote_code": True}
    
    if quantization == "bnb_4bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
    
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()
    
    # 测试不同输入长度
    prompt_lengths = [64, 256, 512, 1024]
    gen_length = 128
    results = []
    
    for plen in prompt_lengths:
        # 构造指定长度的输入
        dummy_text = "请详细解释以下概念：" + " ".join(["人工智能"] * (plen // 4))
        inputs = tokenizer(dummy_text, return_tensors="pt", max_length=plen, truncation=True).to(model.device)
        actual_len = inputs["input_ids"].shape[1]
        
        # Warmup
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        
        # Benchmark (3 runs)
        ttfts, tpots, throughputs = [], [], []
        
        for _ in range(3):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # TTFT: time to first token
            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=1, do_sample=False,
                    return_dict_in_generate=True,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ttft = time.perf_counter() - t0
            ttfts.append(ttft)
            
            # Full generation for TPOT
            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=gen_length, do_sample=False,
                    return_dict_in_generate=True,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_time = time.perf_counter() - t0
            
            n_generated = outputs.sequences.shape[1] - inputs["input_ids"].shape[1]
            tpot = (total_time - ttft) / max(n_generated - 1, 1)
            throughput = n_generated / total_time
            
            tpots.append(tpot)
            throughputs.append(throughput)
        
        result = {
            "prompt_tokens": actual_len,
            "gen_tokens": gen_length,
            "ttft_ms": sum(ttfts) / len(ttfts) * 1000,
            "tpot_ms": sum(tpots) / len(tpots) * 1000,
            "throughput_tps": sum(throughputs) / len(throughputs),
        }
        results.append(result)
        
        print(f"   Input={actual_len:>5} tokens | TTFT={result['ttft_ms']:>7.1f}ms | TPOT={result['tpot_ms']:>6.1f}ms | {result['throughput_tps']:>6.1f} tok/s")
    
    # GPU memory
    if torch.cuda.is_available():
        mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"   Peak GPU memory: {mem_mb:.0f} MB")
    
    return results

def main():
    benchmarks = {}
    
    models_to_test = [
        ("base_bf16", "Qwen/Qwen3-0.6B", None),
        ("dpo_bf16", "outputs/dpo/final", None),
        # ("dpo_bnb4", "outputs/dpo/final", "bnb_4bit"),
        # GPTQ/AWQ 的 benchmark 需要用各自的加载方式
    ]
    
    print("=" * 70)
    print("Inference Benchmark: TTFT / TPOT / Throughput")
    print("=" * 70)
    
    for label, path, quant in models_to_test:
        if not Path(path).exists() and not path.startswith("Qwen/"):
            print(f"⏭️ Skipping {label}")
            continue
        benchmarks[label] = benchmark_hf_model(path, label, quant)
    
    # 保存结果
    with open(RESULTS_DIR / "benchmark.json", "w") as f:
        json.dump(benchmarks, f, indent=2)
    
    print(f"\n✅ Benchmark results saved to {RESULTS_DIR / 'benchmark.json'}")
    
    # --- 对比表 ---
    print("\n" + "=" * 70)
    print("📊 Performance Comparison (prompt=256 tokens)")
    print("=" * 70)
    print(f"{'Model':<15} {'TTFT(ms)':>10} {'TPOT(ms)':>10} {'TPS':>8}")
    print("-" * 48)
    for label, results in benchmarks.items():
        # 取 prompt=256 的结果
        r = results[1] if len(results) > 1 else results[0]
        print(f"{label:<15} {r['ttft_ms']:>10.1f} {r['tpot_ms']:>10.1f} {r['throughput_tps']:>8.1f}")

if __name__ == "__main__":
    main()
    print("\n🎯 All done! Check outputs/eval/ for results.")
