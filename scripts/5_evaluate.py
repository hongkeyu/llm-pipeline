"""
=== Step 5: Evaluation — 全流程评测 ===

全链路位置: Base Model → SFT → DPO → 量化 → [评测]
目的: 用量化指标验证每一步的效果，对比 Base/SFT/DPO/Quantized

评测维度:
  1. Perplexity — 语言建模能力（越低越好）
  2. BLEU — 生成质量（和参考答案的 n-gram 重叠）
  3. Diversity — 生成多样性（避免重复/退化）
  4. Win Rate — DPO 前后的偏好胜率

你需要填写:
  - compute_perplexity(): 从 logits 算困惑度
  - compute_bleu(): BLEU score 完整实现
  - compute_diversity(): distinct n-gram 多样性
  - compute_win_rate(): 比较 DPO 前后的 reward
"""
import json
import math
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config():
    with open("configs/base.yaml") as f:
        return yaml.safe_load(f)


# ============================================================
# 评测指标
# ============================================================

def compute_perplexity(model, tokenizer, texts: list[str], max_length: int = 512) -> float:
    """
    # TODO: Kevin实现

    计算模型在给定文本上的困惑度 (Perplexity)。

    输入:
        model: 语言模型
        tokenizer: 分词器
        texts: list[str] — 测试文本列表
        max_length: 最大 token 长度

    输出:
        ppl: float — 平均困惑度

    Perplexity 定义:
        PPL = exp( -1/N × Σ log P(token_i | token_{<i}) )
        
        其中 N 是总 token 数，log P 是模型给出的 log probability。
        PPL 越低 = 模型越"不惊讶" = 语言建模能力越好。

    计算步骤:
        1. 对每段文本 tokenize
        2. 前向传播得到 logits
        3. 对每个位置，算 log P(真实 token | 前文)
           - logits[:, :-1, :] 对应预测 labels[:, 1:]
           - 用 F.cross_entropy 或手动 log_softmax + gather
        4. 把所有文本的 loss 加权平均（按 token 数加权）
        5. PPL = exp(avg_loss)

    Hint:
        - F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), reduction='sum')
          可以得到该段文本的总 loss
        - 累加所有文本的 total_loss 和 total_tokens
        - PPL = exp(total_loss / total_tokens)
        - 记得 model.eval() + torch.no_grad()

    面试追问:
        - PPL 100 和 PPL 10 差多少？ → 差 10 倍的"不确定性"
        - PPL 的局限？ → 只衡量分布匹配度，不反映生成质量/安全性
        - 为什么不同 tokenizer 的 PPL 不能直接比？ → token 粒度不同，N 不同
    """
    raise NotImplementedError("请实现 compute_perplexity")


def compute_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """
    # TODO: Kevin实现

    计算 BLEU score（从零实现，不用 nltk）。

    输入:
        reference: str — 参考答案（空格分词后的文本）
        hypothesis: str — 模型生成的文本
        max_n: int — 最大 n-gram 阶数（BLEU-4 就是 max_n=4）

    输出:
        bleu: float — BLEU score [0, 1]

    BLEU 公式:
        BLEU = BP × exp( (1/max_n) × Σ_{n=1}^{max_n} log(p_n) )

        其中:
        - p_n = modified n-gram precision
          = (与 reference 匹配的 n-gram 数) / (hypothesis 中 n-gram 总数)
          注意: "modified" 意味着每个 reference n-gram 最多匹配一次（clipped count）
        
        - BP = brevity penalty (惩罚过短的生成)
          = exp(1 - ref_len/hyp_len)  if hyp_len < ref_len
          = 1                          otherwise

    计算步骤:
        1. 分词: ref_tokens = reference.split(), hyp_tokens = hypothesis.split()
        2. 对 n = 1, 2, ..., max_n:
           a. 提取 hypothesis 的所有 n-gram
           b. 提取 reference 的所有 n-gram
           c. 计算 clipped count:
              对每个 hyp n-gram，匹配次数 = min(hyp中出现次数, ref中出现次数)
           d. p_n = clipped_count_sum / hyp_ngram_total
        3. 算 BP
        4. BLEU = BP × exp(mean(log(p_n)))

    Hint:
        - 提取 n-gram: [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        - 用 Counter 统计频次
        - 如果某个 p_n = 0，整个 BLEU = 0（log(0) 问题）
          可以加一个 epsilon 或直接返回 0

    面试追问:
        - BLEU 的局限？ → 只看 n-gram 重叠，不考虑语义；惩罚合理的改写
        - BLEU-1 vs BLEU-4 分别衡量什么？ → BLEU-1 看单词覆盖，BLEU-4 看流畅度
        - 为什么需要 brevity penalty？ → 防止模型只输出高置信度的短句
    """
    raise NotImplementedError("请实现 compute_bleu")


def compute_diversity(texts: list[str]) -> dict:
    """
    # TODO: Kevin实现

    计算生成文本的多样性 (Distinct-N)。

    输入:
        texts: list[str] — 一批生成的文本

    输出:
        diversity: dict — {
            "distinct_1": float,  # unigram 多样性 [0, 1]
            "distinct_2": float,  # bigram 多样性 [0, 1]
            "distinct_3": float,  # trigram 多样性 [0, 1]
        }

    Distinct-N 定义:
        Distinct-N = (不重复的 n-gram 数) / (总 n-gram 数)

        值越高 = 生成越多样；值越低 = 生成越重复/退化

    计算步骤:
        1. 把所有 texts 合并，分词
        2. 对 n = 1, 2, 3:
           a. 提取所有 n-gram
           b. distinct_n = len(set(ngrams)) / len(ngrams)

    Hint:
        - 所有文本先拼成一个 token 列表
        - n-gram 提取和 BLEU 里一样

    面试追问:
        - diversity 和 quality 的 trade-off？
          → temperature 高 → 多样但可能胡说；低 → 保守但重复
        - 怎么同时保证质量和多样性？
          → nucleus sampling (top-p)、typical sampling
    """
    raise NotImplementedError("请实现 compute_diversity")


def compute_win_rate(
    model_a, model_b, tokenizer,
    prompts: list[str], references: list[str],
) -> dict:
    """
    # TODO: Kevin实现

    对比两个模型的生成质量（简单 win rate）。

    输入:
        model_a: 模型 A（例如 SFT 模型）
        model_b: 模型 B（例如 DPO 模型）
        tokenizer: 分词器
        prompts: list[str] — 测试 prompt
        references: list[str] — 参考答案

    输出:
        result: dict — {
            "a_wins": int,
            "b_wins": int,
            "ties": int,
            "b_win_rate": float,  # model_b 的胜率
        }

    评判标准（简化版 — 用 BLEU 作为 proxy）:
        对每个 prompt:
        1. model_a 生成回答 → 算 BLEU(reference, response_a)
        2. model_b 生成回答 → 算 BLEU(reference, response_b)
        3. BLEU 高的获胜

    注意: 真实场景用 GPT-4 做 judge 或人工评测。BLEU 只是 proxy。

    Hint:
        1. 对每个 prompt，用两个模型分别 generate
        2. 调用你实现的 compute_bleu
        3. 比较分数

    面试追问:
        - 为什么 BLEU 不是好的 judge？
          → 不能评估风格、安全性、有用性；语义等价的不同表述会得低分
        - 更好的评测方式？
          → LLM-as-Judge (MT-Bench)、Arena ELO、人工评测
    """
    raise NotImplementedError("请实现 compute_win_rate")


# ============================================================
# 生成辅助函数（已写好）
# ============================================================

def generate_responses(model, tokenizer, prompts, max_new_tokens=256):
    """用模型对每个 prompt 生成回答"""
    device = next(model.parameters()).device
    model.eval()
    responses = []

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=0.7, do_sample=True, top_p=0.9,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        responses.append(response)

    return responses


# ============================================================
# 主流程（已写好）
# ============================================================

def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = cfg["model"]["name"]
    print(f"🚀 Loading base model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    model.eval()

    # 评测文本
    eval_texts = [
        "大语言模型的核心架构是Transformer，它通过自注意力机制实现了对序列数据的高效建模。",
        "机器学习中的过拟合是指模型在训练数据上表现很好，但在新数据上表现差的现象。",
        "深度学习的发展历程中，卷积神经网络和循环神经网络是两个重要的里程碑。",
        "自然语言处理的主要任务包括文本分类、命名实体识别、机器翻译和问答系统。",
        "强化学习通过智能体与环境的交互来学习最优策略。",
    ]

    eval_prompts = [
        "解释什么是注意力机制",
        "LoRA微调的原理是什么",
        "对比监督学习和无监督学习",
        "什么是模型量化？有哪些方法？",
        "DPO和PPO有什么区别？",
    ]

    output_dir = Path(cfg["eval"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # --- 1. Perplexity ---
    print("\n📊 Computing Perplexity...")
    ppl = compute_perplexity(model, tokenizer, eval_texts)
    results["perplexity"] = ppl
    print(f"   PPL: {ppl:.2f}")

    # --- 2. Generation Quality ---
    print("\n📊 Generating responses...")
    responses = generate_responses(model, tokenizer, eval_prompts)

    for i, (p, r) in enumerate(zip(eval_prompts, responses)):
        print(f"   Q: {p}")
        print(f"   A: {r[:100]}...")
        print()

    # --- 3. Diversity ---
    print("📊 Computing Diversity...")
    diversity = compute_diversity(responses)
    results["diversity"] = diversity
    print(f"   Distinct-1: {diversity['distinct_1']:.4f}")
    print(f"   Distinct-2: {diversity['distinct_2']:.4f}")
    print(f"   Distinct-3: {diversity['distinct_3']:.4f}")

    # --- 4. BLEU (self-consistency 作为 proxy) ---
    print("\n📊 Computing BLEU (re-generation consistency)...")
    responses_2 = generate_responses(model, tokenizer, eval_prompts)
    bleu_scores = []
    for r1, r2 in zip(responses, responses_2):
        score = compute_bleu(r1, r2)
        bleu_scores.append(score)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    results["self_bleu"] = avg_bleu
    print(f"   Avg self-BLEU: {avg_bleu:.4f}")

    # 保存
    with open(output_dir / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to {output_dir / 'eval_results.json'}")

    # --- Summary ---
    print(f"\n{'='*50}")
    print("📊 Evaluation Summary")
    print(f"{'='*50}")
    print(f"  Perplexity:  {results['perplexity']:.2f}")
    print(f"  Distinct-1:  {results['diversity']['distinct_1']:.4f}")
    print(f"  Distinct-2:  {results['diversity']['distinct_2']:.4f}")
    print(f"  Self-BLEU:   {results['self_bleu']:.4f}")


if __name__ == "__main__":
    main()
    print("\n🎯 All done! Review outputs/eval/ for full results.")
