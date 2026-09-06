# Part A — Tokenizer Audit & Metric Overhaul

## Executive Recommendation Memo (A4)

**To**: AI Leadership & Infrastructure Engineering Team  
**From**: Senior ML Systems & NLP Audit Engineer  
**Date**: September 6, 2026  
**Subject**: Corrected Tokenizer Fertility Audit, Multi-Language Benchmark, and Routing Decision  

---

### 1. Corrected Headline Numbers

The previous intern's report (`REPORT_v0.md`) claimed that Hindi serving cost is **5.89× to 7.0× higher** than English due to intrinsic properties of the Devanagari script. **This conclusion is invalid.**

Our audit revealed that the high token count was caused by using an English-centric tokenizer (`gpt2`) lacking Indic vocabulary. When evaluated on a 5-language parallel corpus using an Indic-aware multilingual tokenizer (`xlm-roberta-base`), the actual token overhead for Indic traffic is vastly lower:

| Language | Script Family | GPT2 Fertility (`tok/word`) | XLM-R Fertility (`tok/word`) | XLM-R Tokens / Parallel Sentence | Overhead vs. English (XLM-R) |
|---|---|---|---|---|---|
| **English (`eng`)** | Latin | 1.18 | 1.31 | 11.68 tok/sent | 1.00× (Baseline) |
| **Hindi (`hin`)** | Devanagari | 8.39 | 1.45 | 15.40 tok/sent | **1.32×** (vs 5.89× reported) |
| **Tamil (`tam`)** | Dravidian | 26.57 | 2.49 | 18.20 tok/sent | **1.56×** |
| **Kannada (`kan`)** | Dravidian | 24.08 | 2.47 | 16.92 tok/sent | **1.45×** |
| **Telugu (`tel`)** | Dravidian | 22.75 | 2.45 | 17.32 tok/sent | **1.48×** |

*Key Result*: On parallel semantic content, Hindi requires only **1.32× tokens relative to English** under an Indic-aware tokenizer—not 6.0×.

---

### 2. Operational Routing & Metric Recommendation

> [!IMPORTANT]
> **Recommended Single Routing Metric: `tokens / parallel_sentence` (or `tokens / utf8_byte` for byte-level APIs)**

- **Why `tokens / whitespace_word` Fails**: Dravidian languages (Tamil, Kannada, Telugu) are highly agglutinative. A single Dravidian word combines root verbs, tenses, and case markers, carrying the semantic weight of 3–4 English words. Consequently, `tokens / whitespace_word` artificially inflates Dravidian fertility (2.49 tok/word) even when sentence-level token cost (18.20 tok/sent) is nearly identical to Hindi (15.40 tok/sent).
- **Routing Decision**: Route all Indic API requests to models utilizing an Indic-aware tokenizer (such as XLM-RoBERTa, Llama-3, Gemma-2, or Qwen-2.5). Budget an operational serving cost buffer of **+35% for Hindi** and **+50% for Dravidian languages**, rather than the naive 600% budget proposed in v0.

---

### 3. Biggest Production Caveat

> [!WARNING]
> **Code-Switching & Latin Script Mixing in Production Queries**
> 
> Real production traffic in India contains significant "Hinglish" / "Tanglish" and Latin-script mixing (e.g. *"GPU memory full ho gaya, restart kaise karein?"*). Multilingual tokenizers trained strictly on native script corpora can fragment Latin-script Indic words into inefficient subwords. Benchmarks on pure native-script parallel text do not capture code-switching degradation.

---

### 4. Production SLA Counter to Monitor

> [!TIP]
> **Primary SLA Metric**: **`serving:tokens_per_request_by_language`**
> 
> Monitor the exact distribution of input+output tokens per user request broken down by detected language header. Falsification condition: If real production `tokens_per_request` for Hindi exceeds 1.5× English on identical query intents, audit user client inputs for un-normalized NFD Unicode or Latin script code-switching fragmentation.

---

## Detailed Audit of `fertility.py` Flaws (A2)

See [`results/master_evidence_table.csv`](file:///Users/sona/Downloads/Glitchcon_app%202/your-submission/partA/results/master_evidence_table.csv) for full reproducible experimental outputs.

1. **Finding 1 [Code Bug — `line.split(" ")`]**: Naive `split(" ")` retains empty strings on double spaces, artificially inflating word count denominators and under-reporting fertility by **+14.29%**.
2. **Finding 2 [Code Bug — `line.lower()`]**: Lowercasing English text strips proper noun capital letters, reducing GPT-2 subword splits and artificially suppressing the English baseline.
3. **Finding 3 [Code Bug — `len(line)` Code Points]**: Python string length counts UTF-16 code units rather than user-perceived grapheme clusters (`regex \X`). Devanagari combining matras distort character counts by **+36.36%**.
4. **Finding 4 [Conceptual Flaw — Average of Ratios]**: `sum(tok/word)/N` suffers from ratio-estimator bias on short lines. Correct metric is Ratio of Totals ($\frac{\sum \text{tokens}}{\sum \text{words}}$).
5. **Finding 5 [Conceptual Flaw — Script Property Myth]**: Claiming high Hindi fertility is a property of Devanagari script is false; switching to an Indic-aware vocabulary reduces Hindi token count by **-82.98%**.
6. **Finding 6 [Suspicious-but-Correct — NFC Normalization]**: `unicodedata.normalize("NFC", line)` is 100% correct and necessary to prevent decomposed NFD diacritics from splitting into standalone byte tokens.

---

## Script Reproduction Commands

```bash
# 1. Prepare parallel corpus and metadata
python3 partA/scripts/prepare_corpus.py --output_dir partA/corpus

# 2. Run isolated flaw audit on fertility.py
python3 partA/scripts/audit_fertility.py

# 3. Run multi-tokenizer & multi-denominator benchmark
python3 partA/scripts/run_tokenizer_eval.py --corpus_dir partA/corpus
```
