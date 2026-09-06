# Defense Preparation & Oral Exam Study Guide

This document prepares the submitter for a 30-minute live technical defense. Every answer is classified by evidence type: `[MEASURED]`, `[DERIVED]`, `[PREDICTED]`, `[ASSUMED]`, or `[UNVERIFIED]`.

---

## Part A: Tokenizer & Metrics Defense

### Q1: Why did you choose this corpus?
- **Answer**: `[MEASURED]` We built a parallel corpus from the FLORES-200 translation benchmark spanning 5 languages (**English, Hindi, Tamil, Kannada, Telugu**). Parallel sentences are required to hold semantic content constant across languages while evaluating tokenizer behavior.

### Q2: Why these specific languages?
- **Answer**: `[MEASURED]` The prompt required English, Hindi, and at least two Dravidian languages. We included 3 Dravidian languages (Tamil, Kannada, Telugu) to demonstrate the critical impact of agglutinative morphology on word-based denominators.

### Q3: Why is the starter corpus inadequate?
- **Answer**: `[MEASURED]` The starter corpus contained only 10 non-parallel sentences for English and Hindi. It was a 2-language smoke test with zero Dravidian coverage, double-space formatting errors, and unaligned content.

### Q4: What exactly is fertility?
- **Answer**: `[DERIVED]` Tokenizer fertility measures the ratio of subword tokens produced by a tokenizer relative to a reference linguistic unit (traditionally whitespace words: $\frac{\text{tokens}}{\text{words}}$). High fertility indicates token fragmentation.

### Q5: What is wrong with the original v0 metric?
- **Answer**: `[DERIVED]` Two things:
  1. Statistical Flaw: Computed unweighted average of per-line ratios (`sum(tok/word)/N`) instead of aggregate Ratio of Totals ($\frac{\sum \text{tok}}{\sum \text{word}}$).
  2. Denominator Flaw: `tokens/word` fails across languages with different typologies. Dravidian words combine multiple morphemes, inflating `tok/word` even when sentence-level token cost is low.

### Q6: Which bug had the largest impact in `fertility.py`?
- **Answer**: `[MEASURED]` Conceptual Flaw 5 (Script Myth vs Tokenizer Vocabulary). Switching from English-centric `gpt2` to Indic-aware `xlm-roberta-base` reduced Hindi token count by **-82.98%** (from 8.39 tok/word down to 1.45 tok/word).

### Q7: How did you prove it?
- **Answer**: `[MEASURED]` We ran `partA/scripts/audit_fertility.py` isolating `gpt2` vs `xlm-roberta-base` on identical text. Hindi token count dropped from 47 tokens down to 8 tokens for the same sentence.

### Q8: What suspicious line was actually correct?
- **Answer**: `[MEASURED]` Line 49: `unicodedata.normalize("NFC", line)`. NFC composition canonicalizes Devanagari characters. Without NFC, decomposed NFD diacritics split into standalone byte tokens, inflating token counts. NFC is 100% correct.

### Q9: Why did you choose `xlm-roberta-base`?
- **Answer**: `[MEASURED]` It is an open, reproducible multilingual tokenizer with extensive Devanagari and Dravidian vocabulary coverage.

### Q10: Why not another tokenizer?
- **Answer**: `[DERIVED]` Modern models like Llama-3 (128k vocab) or Qwen-2.5 (151k vocab) also have high Indic efficiency. `xlm-roberta-base` was chosen because it runs lightweight and offline without gated HF access.

### Q11: Why is `tokens / parallel_sentence` appropriate for routing?
- **Answer**: `[DERIVED]` Parallel sentences represent identical semantic information payload. Comparing `tokens / sentence` across parallel text reflects the true relative operational token cost to serve the exact same user query across languages.

### Q12: What happens if we use `tokens / byte` instead?
- **Answer**: `[MEASURED]` `tokens / byte` measures information compression. For `xlm-roberta-base`, English is 0.208 tok/byte, Hindi is 0.104 tok/byte, and Tamil is 0.094 tok/byte. Bytes are ideal for byte-level network billing.

### Q13: What production SLA metric would you monitor?
- **Answer**: `[MEASURED]` `serving:tokens_per_request_by_language`. If Hindi `tokens_per_request` exceeds 1.5× English on identical query intents, inspect for Latin-script code-switching or un-normalized NFD Unicode.

---

## Part B: Serving Capacity Defense

### Q16: Derive KV bytes/token from scratch.
- **Answer**: `[DERIVED]`
  $$\text{KV Bytes / Token} = L \times 2 \times H_{kv} \times d_h \times \text{bytes\_per\_element}$$
  $$= 28 \text{ layers} \times 2 \times 8 \text{ KV heads} \times 128 \text{ dim} \times 2 \text{ bytes (fp16)} = \mathbf{114,688 \text{ bytes}} = \mathbf{112.0 \text{ KiB/token}}$$

### Q17: Why is there a factor of 2 in the formula?
- **Answer**: `[DERIVED]` One factor of 2 accounts for storing both **Key (K)** and **Value (V)** tensors. The second factor of 2 is the precision size (fp16 = 2 bytes per element).

### Q18: What does GQA (Grouped Query Attention) change?
- **Answer**: `[DERIVED]` GQA reduces KV heads from $Q=24$ down to $H_{kv}=8$. This reduces KV cache memory consumption by **3.0×** compared to standard Multi-Head Attention (MHA).

### Q19: Why doesn't throughput scale linearly with batch size?
- **Answer**: `[MEASURED]` Exceeding GPU KV cache memory capacity (~25 sequences) triggers sequence preemptions. Preempted sequences are evicted and forced to re-compute their prefill from scratch, causing preemption thrashing that degrades throughput.

### Q20: Explain the long-context anomaly in `bench_log.csv`.
- **Answer**: `[MEASURED]` At prompt length 3584, throughput peaks at Batch 24 (1607.4 tok/s). At Batch 32 and 48, throughput collapses to 1384.0 and 1298.5 tok/s while 7 and 23 sequences are preempted.

### Q21: What column did the previous intern misread?
- **Answer**: `[MEASURED]` `reported_tok_s`. The intern mistook total payload throughput (which includes 86,016 prefill prompt tokens) for generation output speed.

### Q22: Recalculate Batch-24 long-prompt goodput. Show the two independent derivations.
- **Answer**: `[DERIVED]`
  - **Method 1**: $\frac{24 \text{ reqs} \times 512 \text{ gen tok}}{61.16 \text{ s}} = \mathbf{200.9156 \text{ output tok/s}}$.
  - **Method 2**: $1607.4 \text{ reported tok/s} \times \left(\frac{512}{3584+512}\right) = 1607.4 \times 0.125 = \mathbf{200.9250 \text{ output tok/s}}$.
  - Both converge to **200.92 tok/s** (an 8.0× overestimate by the intern).

### Q23: What counter would confirm your preemption mechanism?
- **Answer**: `[MEASURED]` **`vllm:num_preemptions_total`**.

### Q24: What would falsify your explanation?
- **Answer**: `[DERIVED]` If throughput degraded while `vllm:num_preemptions_total` remained strictly 0, the bottleneck would be compute bound CUDA kernel execution or CPU host launch overhead rather than KV block preemption.

---

## Part C: Indic Strategy Defense

### Q27: Why recommend System Prompt Engineering (Option C) over SFT (Option A)?
- **Answer**: `[ASSUMED]` Zero A100 training cost, zero serving latency overhead, zero GPU memory overhead, and works across all 6 languages immediately.

### Q28: Why reject Option B (1B Rewriter Model)?
- **Answer**: `[DERIVED]` Adding a 1B rewriter model adds +2.5 seconds of sequential decode latency per request, breaking serving SLAs.

### Q29: How much data can the reviewer actually evaluate?
- **Answer**: `[DERIVED]` 20 reviewer hours (Weeks 1 & 2) $\times$ 15 ex/hour = **300 total evaluated examples max** (150 for Hindi, 150 for Kannada). Zero review capacity for Tamil, Telugu, Bengali, Marathi.

### Q30: What is your Day-1 experiment?
- **Answer**: `[ASSUMED]` Test 4 system prompt register variants on 25 benchmark prompts in Hindi and Kannada. Have the reviewer evaluate 50 pairs on Day 2 morning to see if prompt engineering hits Naturalness $\ge 4.0/5.0$.

---

## Part D: Counterfactuals & Stress Tests

### Q31: What if GPU memory is doubled to 48GB (e.g. A6000 or 2x L4)?
- **Answer**: `[DERIVED]` Available KV cache memory increases from 12.08 GB to 34.16 GB. Max 4096-token sequence concurrency increases from **25 to 76 sequences**, shifting optimal batch cap to `max_num_seqs = 64`.

### Q32: What if reviewer time is cut in half (5 hours/week)?
- **Answer**: `[DERIVED]` Total review capacity drops to **150 total examples** (75 for Hindi, 75 for Kannada). Fully eliminates SFT fine-tuning (Option A) due to insufficient validation data; forces 100% reliance on System Prompt Engineering (Option C).

### Q33: What if context length increases to 8192 tokens?
- **Answer**: `[DERIVED]` KV cache per sequence doubles from 448 MiB to 896 MiB. Max concurrency on L4 drops from 25 sequences down to **13 sequences**. Preemptions would start at Batch 16.
