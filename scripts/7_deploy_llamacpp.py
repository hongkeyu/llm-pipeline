"""
Step 7: Deploy with llama.cpp (GGUF)
适合 CPU / 边缘设备（如 Jetson Orin Nano）部署
"""
import subprocess
import time
import json
from pathlib import Path

GGUF_DIR = Path("outputs/quantized/gguf")
GGUF_DIR.mkdir(parents=True, exist_ok=True)

def convert_to_gguf(model_path: str, output_path: str = None):
    """
    HF 模型 → GGUF 格式
    
    面试要点:
    - GGUF 是 llama.cpp 的原生格式，支持 CPU 推理
    - 量化类型: Q4_K_M (推荐), Q5_K_M (更高精度), Q8_0 (最高精度)
    - K-quant: 对不同层用不同 bit 数，重要层保留更高精度
    """
    if output_path is None:
        output_path = str(GGUF_DIR / "model-f16.gguf")
    
    print("📦 Converting to GGUF format...")
    
    # Step 1: 转换为 GGUF
    cmd = [
        "python", "llama.cpp/convert_hf_to_gguf.py",
        model_path,
        "--outfile", output_path,
        "--outtype", "f16",
    ]
    subprocess.run(cmd, check=True)
    print(f"✅ GGUF F16 saved to {output_path}")
    
    return output_path

def quantize_gguf(input_gguf: str, quant_types: list = None):
    """
    GGUF 量化
    
    面试要点:
    - Q4_K_M: 4-bit with K-quant, Medium quality → 最常用
    - Q5_K_M: 5-bit, 精度更高，略大
    - Q8_0: 8-bit, 几乎无损
    - 重要性矩阵 (imatrix): 可以进一步优化量化质量
    """
    if quant_types is None:
        quant_types = ["Q4_K_M", "Q5_K_M", "Q8_0"]
    
    results = []
    for qtype in quant_types:
        output = str(GGUF_DIR / f"model-{qtype}.gguf")
        print(f"\n🔧 Quantizing to {qtype}...")
        
        t0 = time.time()
        cmd = ["llama.cpp/llama-quantize", input_gguf, output, qtype]
        subprocess.run(cmd, check=True)
        quant_time = time.time() - t0
        
        size_mb = Path(output).stat().st_size / (1024 * 1024)
        results.append({
            "quant_type": qtype,
            "size_mb": size_mb,
            "quant_time_s": quant_time,
            "path": output,
        })
        print(f"✅ {qtype}: {size_mb:.1f} MB ({quant_time:.1f}s)")
    
    return results

def benchmark_llamacpp(gguf_path: str, n_threads: int = 4):
    """
    llama.cpp 推理性能测试
    
    面试要点:
    - llama-bench 测 token 生成速度
    - pp (prompt processing) = prefill 速度 → 影响 TTFT
    - tg (text generation) = decode 速度 → 影响 TPOT
    """
    print(f"\n📊 Benchmarking {Path(gguf_path).name}...")
    
    cmd = [
        "llama.cpp/llama-bench",
        "-m", gguf_path,
        "-t", str(n_threads),
        "-p", "512",    # prompt length for pp test
        "-n", "128",    # tokens to generate for tg test
        "-r", "3",      # repeat 3 times
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    return result.stdout

def interactive_chat(gguf_path: str, n_threads: int = 4):
    """交互式聊天"""
    cmd = [
        "llama.cpp/llama-cli",
        "-m", gguf_path,
        "-t", str(n_threads),
        "-n", "256",
        "-c", "2048",
        "--interactive-first",
        "-p", "You are a helpful assistant.",
    ]
    print(f"💬 Starting interactive chat with {Path(gguf_path).name}...")
    subprocess.run(cmd)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/dpo/final", help="HF model path")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat")
    args = parser.parse_args()
    
    if not args.skip_convert:
        f16_path = convert_to_gguf(args.model)
        results = quantize_gguf(f16_path)
        
        # Benchmark all quantized models
        for r in results:
            benchmark_llamacpp(r["path"])
    
    if args.chat:
        q4_path = str(GGUF_DIR / "model-Q4_K_M.gguf")
        interactive_chat(q4_path)
