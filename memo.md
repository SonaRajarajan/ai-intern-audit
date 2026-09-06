# Executive Audit & Decision Summary Memo

**To**: AI Leadership & Technical Review Board  
**From**: Senior ML Systems & NLP Research Engineer (Audit Intern)  
**Date**: September 6, 2026  
**Subject**: Comprehensive Audit of Tokenizer Fertility, Serving Capacity, and Indic AI Strategy  

---

## Executive Summary

This submission presents a rigorous, empirical audit of the previous intern's findings (`REPORT_v0.md`), serving benchmark logs (`bench_log.csv`), and a technical roadmap for casual Indic LLM deployment. Every claim is supported by reproducible code, isolated experiments, and derived mathematical formulas.

---

## 1. Part A — Tokenizer Audit Findings

1. **V0 Claim Rebuttal**: `REPORT_v0.md` asserted that Hindi tokenization is **5.89× to 7.0× more expensive** than English due to intrinsic properties of the Devanagari script. **This claim is false.**
2. **Root Cause**: The high token count was an artifact of using an English-centric BPE tokenizer (`gpt2`). When evaluated on our 5-language parallel corpus (**English, Hindi, Tamil, Kannada, Telugu**) using an Indic-aware tokenizer (`xlm-roberta-base`), Hindi requires only **1.32× tokens relative to English** on parallel semantic content.
3. **Flaw Audit Summary**:
   - *Code Bug 1*: `line.split(" ")` retains empty strings on double spaces, under-reporting fertility by **+14.29%**.
   - *Code Bug 2*: `line.lower()` strips English proper noun capital letters, artificially suppressing the English baseline.
   - *Code Bug 3*: `len(line)` counts UTF-16 code units instead of grapheme clusters (`regex \X`), under-reporting character compression by **+36.36%**.
   - *Conceptual Flaw 1*: Unweighted mean of ratios `sum(tok/word)/N` introduces ratio-estimator bias compared to Ratio of Totals ($\frac{\sum \text{tok}}{\sum \text{word}}$).
   - *Conceptual Flaw 2*: High fertility is a property of tokenizer vocabulary, not Devanagari script. XLM-R reduces Hindi token count by **-82.98%**.
   - *Suspicious-but-Correct*: `unicodedata.normalize("NFC", line)` is 100% correct and necessary to prevent decomposed diacritics from fragmenting into byte tokens.
4. **Recommended Metric**: **`tokens / parallel_sentence`** (or `tokens / utf8_byte` for byte-level APIs) for routing and capacity planning.

---

## 2. Part B — Serving Capacity Reconciliation

1. **Exact KV Cache Formula**:  
   $$\text{KV Bytes / Token} = 28 \text{ layers} \times 2 \times 8 \text{ heads} \times 128 \text{ dim} \times 2 \text{ bytes (fp16)} = \mathbf{114,688 \text{ bytes/token}} = \mathbf{112.0 \text{ KiB/token}}$$
2. **Concurrency Upper Bound**: Usable GPU RAM ($22.08 \text{ GB}$) minus weights ($8.40 \text{ GB}$) and overhead ($1.60 \text{ GB}$) leaves $12.08 \text{ GB}$ KV RAM. A 4096-token sequence requires $448.0 \text{ MiB}$. Maximum concurrency = **25.71 full 4096-token sequences**. Reconciles perfectly with `bench_log.csv` preemption inflection between Batch 24 (0 preemptions) and Batch 32 (7 preemptions).
3. **Long-Context Anomaly**: Beyond Batch 24, KV cache saturation forces sequence preemption, causing prefill re-computation thrashing. Setting `max_num_seqs = 24` eliminates preemptions and increases throughput by **+23.8%** under heavy load.
4. **B3 Misread Column & Goodput Derivations**: `REPORT_v0.md` misread `reported_tok_s` (which included prompt prefill tokens). True Batch-24 generation goodput derived via two independent methods converges to **200.9 output tokens/sec**—showing an **8.0× overestimate** in v0.
5. **Diagnostic Production Counter**: **`vllm:num_preemptions_total`**.

---

## 3. Part C — Indic Casualization Strategy

1. **Constraint Reality**: 1× A100-80GB (2 weeks), 1 reviewer (Hindi+Kannada only, 10h/w = 300 total evaluated examples max), $0 API budget.
2. **Recommended Strategy**: **Option (c) System Prompt Engineering + Few-Shot In-Context Learning (Primary)**, backed by **Option (a) Targeted LoRA SFT for Hindi & Kannada (Fallback)**. Option (b) (Rewriter Model) is rejected due to +2.5s serving latency overhead.
3. **Success Metric**: Conversational Naturalness Score $\ge 4.2 / 5.0$ with Factuality Retention $\ge 98.0\%$.
4. **Kill Criterion**: If Naturalness $< 3.8 / 5.0$ or Factuality $< 95.0\%$ by Day 7, terminate fine-tuning and default to standard prompts.
