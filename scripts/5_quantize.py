"""
Step 5: Quantization
对比 GPTQ / AWQ / bitsandbytes 量化方案
"""
import yaml
import json
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS_DIR = Path("outputs/quantized")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_config():
    with open("configs/quant_config.yaml") as f:
        return yaml.safe_load(f)

def get_model_size_mb(model_path: str) -> float:
    """计算模型文件大小"""
    p = Path(model_path)
    total = sum(f.stat().st_size for f in p.rglob("*.safetensors"))
    if total == 0:
        total = sum(f.stat().st_size for f in p.rglob("*.bin"))
    return total / (1024 * 1024)

def quantize_gptq(model_path: str, bits: int, output_dir: str):
    """GPTQ 量化"""
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    
    print(f"🔧 GPTQ-INT{bits} quantizing...")
    
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=128,
        desc_act=True,
        damp_percent=0.01,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoGPTQForCausalLM.from_pretrained(
        model_path,
        quantize_config,
        trust_remote_code=True,
    )
    
    # 校准数据
    from datasets import load_dataset
    calib_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    calib_texts = [text for text in calib_ds["text"] if len(text) > 50][:128]
    calib_data = [tokenizer(text, return_tensors="pt", max_length=512, truncation=True) for text in calib_texts]
    
    t0 = time.time()
    model.quantize(calib_data)
    quant_time = time.time() - t0
    
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    size_mb = get_model_size_mb(output_dir)
    print(f"✅ GPTQ-INT{bits}: {size_mb:.1f} MB, took {quant_time:.1f}s")
    return {"method": f"GPTQ-INT{bits}", "size_mb": size_mb, "quant_time_s": quant_time}

def quantize_awq(model_path: str, output_dir: str):
    """AWQ 量化"""
    from awq import AutoAWQForCausalLM
    
    print("🔧 AWQ-INT4 quantizing...")
    
    model = AutoAWQForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }
    
    t0 = time.time()
    model.quantize(tokenizer, quant_config=quant_config)
    quant_time = time.time() - t0
    
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    size_mb = get_model_size_mb(output_dir)
    print(f"✅ AWQ-INT4: {size_mb:.1f} MB, took {quant_time:.1f}s")
    return {"method": "AWQ-INT4", "size_mb": size_mb, "quant_time_s": quant_time}

def quantize_bnb(model_path: str, bits: int = 4):
    """bitsandbytes NF4 量化 (运行时量化，不保存)"""
    from transformers import BitsAndBytesConfig
    
    print(f"🔧 BnB-NF{bits} loading...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True if bits == 4 else False,
        load_in_8bit=True if bits == 8 else False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    load_time = time.time() - t0
    
    # 估算显存
    mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
    print(f"✅ BnB-NF{bits}: GPU mem {mem_mb:.1f} MB, load time {load_time:.1f}s")
    return {"method": f"BnB-NF{bits}", "gpu_mem_mb": mem_mb, "load_time_s": load_time}

def compute_perplexity(model_path: str, is_gptq: bool = False, is_awq: bool = False):
    """计算量化后模型的困惑度"""
    from datasets import load_dataset
    import math
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    if is_gptq:
        from auto_gptq import AutoGPTQForCausalLM
        model = AutoGPTQForCausalLM.from_quantized(model_path, device_map="auto", trust_remote_code=True)
    elif is_awq:
        from awq import AutoAWQForCausalLM
        model = AutoAWQForCausalLM.from_quantized(model_path, fuse_layers=False, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
    
    # WikiText-2 perplexity
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    
    with torch.no_grad():
        outputs = model(encodings.input_ids.to(model.device), labels=encodings.input_ids.to(model.device))
    
    ppl = math.exp(outputs.loss.item())
    return ppl

def main():
    cfg = load_config()
    model_path = cfg["model"]["name"]
    results = []
    
    print("=" * 60)
    print("Quantization Comparison")
    print("=" * 60)
    
    # 1. Baseline (FP16/BF16)
    print("\n📏 Baseline (BF16)...")
    baseline_size = get_model_size_mb(model_path)
    baseline_ppl = compute_perplexity(model_path)
    results.append({"method": "BF16 (baseline)", "size_mb": baseline_size, "perplexity": baseline_ppl})
    print(f"   Size: {baseline_size:.1f} MB, PPL: {baseline_ppl:.2f}")
    
    # 2. GPTQ INT4 & INT8
    for bits in [8, 4]:
        out_dir = str(RESULTS_DIR / f"gptq-int{bits}")
        info = quantize_gptq(model_path, bits, out_dir)
        info["perplexity"] = compute_perplexity(out_dir, is_gptq=True)
        results.append(info)
    
    # 3. AWQ INT4
    out_dir = str(RESULTS_DIR / "awq-int4")
    info = quantize_awq(model_path, out_dir)
    info["perplexity"] = compute_perplexity(out_dir, is_awq=True)
    results.append(info)
    
    # 4. BnB NF4
    info = quantize_bnb(model_path, bits=4)
    results.append(info)
    
    # --- Summary ---
    print("\n" + "=" * 60)
    print("📊 Quantization Results Summary")
    print("=" * 60)
    print(f"{'Method':<20} {'Size (MB)':>10} {'PPL':>8} {'Δ PPL':>8}")
    print("-" * 50)
    for r in results:
        ppl = r.get("perplexity", "N/A")
        delta = f"+{ppl - baseline_ppl:.2f}" if isinstance(ppl, float) else "N/A"
        ppl_str = f"{ppl:.2f}" if isinstance(ppl, float) else ppl
        print(f"{r['method']:<20} {r.get('size_mb', 'N/A'):>10} {ppl_str:>8} {delta:>8}")
    
    # 保存结果
    with open(RESULTS_DIR / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {RESULTS_DIR / 'comparison.json'}")

if __name__ == "__main__":
    main()
    print("\n🎯 Quantization done! Next: python scripts/6_deploy_vllm.py")
