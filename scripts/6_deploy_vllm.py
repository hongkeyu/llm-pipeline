"""
Step 6: Deploy with vLLM
启动 vLLM 推理服务，支持 OpenAI-compatible API
"""
import subprocess
import time
import json
import requests

def start_vllm_server(
    model_path: str = "outputs/dpo/final",
    port: int = 8000,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 2048,
    quantization: str = None,
):
    """
    启动 vLLM OpenAI-compatible server
    
    面试要点:
    - PagedAttention: 将 KV Cache 分页管理，类似 OS 虚拟内存，避免碎片化
    - Continuous Batching: 请求级动态合批，不等最长序列完成
    - Prefix Caching: 相同 system prompt 的 KV Cache 可复用
    """
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--dtype", "bfloat16",
        "--enable-prefix-caching",  # 开启 prefix caching
    ]
    
    if quantization:
        cmd.extend(["--quantization", quantization])
    
    print(f"🚀 Starting vLLM server on port {port}...")
    print(f"   Model: {model_path}")
    print(f"   Command: {' '.join(cmd)}")
    
    proc = subprocess.Popen(cmd)
    
    # 等待服务就绪
    for i in range(60):
        try:
            r = requests.get(f"http://localhost:{port}/health")
            if r.status_code == 200:
                print(f"✅ vLLM server ready! (took {i}s)")
                return proc
        except requests.ConnectionError:
            pass
        time.sleep(1)
    
    print("❌ Server failed to start within 60s")
    proc.kill()
    return None

def test_inference(port: int = 8000, model_path: str = "outputs/dpo/final"):
    """测试推理"""
    url = f"http://localhost:{port}/v1/chat/completions"
    
    test_cases = [
        "请解释什么是大模型的量化？为什么需要量化？",
        "写一个Python快速排序算法",
        "对比DPO和PPO的优缺点",
    ]
    
    results = []
    for prompt in test_cases:
        payload = {
            "model": model_path,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.7,
        }
        
        t0 = time.time()
        r = requests.post(url, json=payload)
        latency = time.time() - t0
        
        if r.status_code == 200:
            data = r.json()
            response = data["choices"][0]["message"]["content"]
            usage = data["usage"]
            
            ttft = data.get("timings", {}).get("time_to_first_token", None)
            
            results.append({
                "prompt": prompt[:50],
                "response_len": usage["completion_tokens"],
                "latency_s": latency,
                "tokens_per_sec": usage["completion_tokens"] / latency,
            })
            
            print(f"\n📝 Q: {prompt[:60]}...")
            print(f"   A: {response[:150]}...")
            print(f"   Tokens: {usage['completion_tokens']}, Latency: {latency:.2f}s, TPS: {usage['completion_tokens']/latency:.1f}")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/dpo/final")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--quant", default=None, help="gptq, awq, or None")
    args = parser.parse_args()
    
    proc = start_vllm_server(args.model, args.port, quantization=args.quant)
    if proc:
        try:
            test_inference(args.port, args.model)
            input("\n按 Enter 停止服务...")
        finally:
            proc.terminate()
            proc.wait()
