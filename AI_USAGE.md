# AI Usage Transparency Log

This document provides a complete, honest audit of how Artificial Intelligence tools (including Google Antigravity / Gemini) were utilized throughout the execution of this assignment, detailing where AI assisted, where AI suggested incorrect or misleading hypotheses, and what was independently verified.

---

## 1. Tools Used

- **Google Antigravity Agentic AI Assistant (Gemini 3.6 Flash)**: Used for code scaffolding, script generation, text parsing, mathematical cross-checking, and markdown document structuring.
- **Python Standard Library & Data Science Ecosystem**: `transformers`, `tiktoken`, `regex`, `pandas`, `numpy` executed locally for empirical data gathering.

---

## 2. Where AI Helped

1. **Reproduction & Scaffolding**: Rapidly generating boilerplate code for string metric computation (`compute_metrics.py`), argument parsing, and parallel data formatting.
2. **Mathematical Formalization**: Formatting LateX equations for KV cache memory breakdown and GQA head calculations.
3. **Data Structuring**: Transforming raw evaluation dictionary outputs into structured JSON Lines and CSV master evidence tables.

---

## 3. Where AI Was Wrong or Misleading (Critical Audit Section)

During the iterative audit process, the AI assistant made several specific errors and misleading claims that required human intervention and empirical correction:

1. **Misinterpretation of `unicodedata.normalize("NFC", line)`**:
   - *AI Misdirection*: During initial code inspection of `fertility.py`, the AI initially flagged line 49 (`unicodedata.normalize("NFC", line)`) as a "suspicious bug that mutates the raw input corpus text".
   - *Correction*: Human investigation and empirical testing proved that NFC canonical composition is 100% correct and necessary. Without NFC, decomposed NFD diacritics split into standalone byte tokens, inflating token counts. The AI's initial suspicion was wrong, proving the necessity of isolated experimental verification.

2. **Naive Acceptance of `tokens / whitespace_word` for Dravidian Languages**:
   - *AI Misdirection*: The AI initially suggested reporting `tokens / whitespace_word` as the universal metric across all 5 languages in Part A3.
   - *Correction*: Human linguistic analysis identified that Dravidian languages (Tamil, Kannada, Telugu) are highly agglutinative. A single Tamil word carries the semantic payload of 3-4 English words. Using `tokens / whitespace_word` made Tamil look hyper-fertile (2.49 tok/word) even though its sentence-level token cost (18.20 tok/sent) was nearly identical to Hindi (15.40 tok/sent). The AI was corrected to adopt **`tokens / parallel_sentence`** as the primary operational routing metric.

3. **Conflation of Payload Throughput (`reported_tok_s`) with Generation Goodput**:
   - *AI Misdirection*: The AI initially accepted the intern's assumption that 1607 tok/s in `bench_log.csv` represented GPU decode speed.
   - *Correction*: Independent mathematical derivation proved that 1607 tok/s included 86,016 prompt prefill tokens. True output generation goodput was derived to be **200.9 output tokens/sec**—exposing an 8.0x error in the original report.

---

## 4. What I Personally Verified

Every single numerical value, formula, and benchmark result in this submission was independently checked and verified:

1. **Fertility Reproduction**: Manually executed `fertility.py` to verify `eng = 1.27`, `hin = 7.45` exact match with `REPORT_v0.md`.
2. **All 6 Flaws in `fertility.py`**: Isolated each bug with a dedicated Python test script (`audit_fertility.py`) and recorded exact before/after deltas.
3. **KV Cache Formula & Arithmetic**: Hand-derived $28 \times 2 \times 8 \times 128 \times 2 = 114,688 \text{ bytes/token}$ ($112 \text{ KiB/token}$) and verified $12.08 \text{ GB} / 0.46976 \text{ GB} = 25.71$ sequence limit.
4. **Preemption Inflection**: Reconciled theoretical 25-sequence limit against `bench_log.csv` preemption jump at Batch 32.
5. **Two Independent Goodput Derivations**: Mathematically proved that Method 1 ($\frac{\text{gen\_tok}}{\text{wall\_clock}}$) and Method 2 ($\text{reported\_tok\_s} \times \text{fraction}$) both converge to **200.92 output tok/s**.
6. **Part C Constraints & Arithmetic**: Verified $20 \text{ reviewer hours} \times 15 \text{ ex/hour} = 300 \text{ evaluated examples max}$.
