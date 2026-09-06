# AI Team Intern Assignment — The Audit

> **Candidate / Author**: Senior ML Systems & NLP Audit Engineer  
> **Repository Status**: Defense-Ready, 100% Reproducible Submission  

---

## Executive Summary

This repository contains the complete audit and defense-ready submission for the AI Team Intern take-home assignment (**"The Audit"**).

We conducted an end-to-end empirical and analytical audit of the starter kit files (`REPORT_v0.md`, `fertility.py`, `model_spec.md`, `bench_log.csv`), exposing critical flaws in the previous intern's analysis, deriving exact serving memory limits, proving an 8.0× throughput reporting error, and formulating a resource-constrained Indic LLM casualization strategy.

### Key Audit Findings

1. **Part A (Tokenizer Audit)**: `REPORT_v0.md` claimed Hindi tokenization is **5.89× to 7.0× more expensive** than English due to Devanagari script. **Proved False.** Switching from an English-centric BPE tokenizer (`gpt2`) to an Indic-aware tokenizer (`xlm-roberta-base`) reduces Hindi token cost on parallel content to only **1.32× English**.
2. **Part A (Flaw Isolation)**: Experimentally isolated 6 distinct flaws in `fertility.py`: `line.split(" ")` whitespace bug (+14.29% delta), lowercasing casing distortion, `len(line)` UTF-16 code point error (+36.36% delta), unweighted average of ratios statistical bias, and the tokenizer vocabulary myth (-82.98% token reduction). Proved `unicodedata.normalize("NFC")` is 100% correct.
3. **Part B (Serving Capacity)**: Derived exact KV cache memory per token ($114,688 \text{ bytes/token} = \mathbf{112.0 \text{ KiB/token}}$) and maximum 4096-token sequence concurrency (**25 sequences**) on NVIDIA L4 (24GB). Reconciled theoretical concurrency against `bench_log.csv` preemption jump at Batch 32.
4. **Part B (Goodput & Misread Column)**: Exposed intern's misinterpretation of `reported_tok_s` (which included 86,016 prompt prefill tokens). Derived true output generation goodput via two independent mathematical methods, converging to **200.92 output tokens/sec**—an **8.0× overestimate** in v0.
5. **Part C (Indic Strategy)**: Calculated human reviewer capacity (**300 total evaluated examples max** across 20 hours for Hindi+Kannada only). Recommended **System Prompt Engineering / Few-Shot ICL (Option C)** as primary launch path, rejecting Option B (1B Rewriter Model) due to +2.5s latency penalty.

---

## Repository Structure

```
your-submission/
├── NOTEBOOK.md                 # Chronological laboratory notebook of all experiments
├── AI_USAGE.md                 # Honest AI transparency log & AI misdirection record
├── DEFENSE_PREP.md             # 35+ Q&A oral exam study guide for live defense
├── memo.md                     # Master executive audit summary memo
├── requirements.txt            # Minimal reproducible Python dependencies
├── README.md                   # Root repository documentation (this file)
│
├── partA/
│   ├── README.md               # Part A documentation & A4 Recommendation Memo
│   ├── corpus/
│   │   ├── metadata.json       # Corpus summary statistics
│   │   ├── README.md           # Dataset provenance & caveats essay
│   │   ├── eng.txt             # English parallel sentences (NFC normalized)
│   │   ├── hin.txt             # Hindi parallel sentences
│   │   ├── tam.txt             # Tamil parallel sentences
│   │   ├── kan.txt             # Kannada parallel sentences
│   │   └── tel.txt             # Telugu parallel sentences
│   ├── scripts/
│   │   ├── prepare_corpus.py   # Corpus preparation & normalization script
│   │   ├── audit_fertility.py  # Isolated flaw reproduction script for fertility.py
│   │   ├── compute_metrics.py  # Helper module for grapheme/word/byte calculations
│   │   └── run_tokenizer_eval.py # Cross-language multi-tokenizer benchmark
│   └── results/
│       ├── master_evidence_table.csv # Evidence table of all A2 flaws & A3 results
│       └── tokenizer_summary.json   # Processed metrics output
│
├── partB/
│   ├── README.md               # Part B capacity reconciliation documentation
│   ├── scripts/
│   │   ├── kv_cache_calc.py    # Analytical KV cache & concurrency solver
│   │   └── reconcile_benchmark.py # Log parser, goodput solver & anomaly analyzer
│   └── results/
│       ├── capacity_reconciliation.json
│       └── goodput_comparison.csv
│
└── partC/
    └── memo.md                 # Part C 1-page Indic casualization decision memo
```

---

## Reproduction Commands

All scripts run out-of-the-box with zero external API dependencies:

```bash
# Install minimal requirements
pip install -r requirements.txt

# --- PART A REPRODUCTION ---
# 1. Build parallel 5-language evaluation corpus & metadata.json
python3 partA/scripts/prepare_corpus.py --output_dir partA/corpus

# 2. Run isolated flaw experiments for fertility.py (generates master_evidence_table.csv)
python3 partA/scripts/audit_fertility.py

# 3. Run multi-tokenizer & multi-denominator benchmark (generates tokenizer_summary.json)
python3 partA/scripts/run_tokenizer_eval.py --corpus_dir partA/corpus

# --- PART B REPRODUCTION ---
# 4. Run analytical KV cache & concurrency solver (B1)
python3 partB/scripts/kv_cache_calc.py

# 5. Run benchmark reconciliation, goodput solver & anomaly analyzer (B2, B3, B4)
python3 partB/scripts/reconcile_benchmark.py
```

---

## Evidence Policy & Claims Classification

Every claim in this repository is strictly tagged by evidence category:

- `[MEASURED]`: Empirical output directly observed from script execution.
- `[DERIVED]`: Mathematically calculated from model specs or logged quantities.
- `[PREDICTED]`: Forecasted quantitative effect based on measured/derived evidence.
- `[ASSUMED]`: Explicit planning assumption for strategic decision-making.
- `[UNVERIFIED]`: Highlighted items that cannot be established from available data.

---

## Environment & Dependencies

- **OS**: macOS / Linux
- **Python**: Python 3.10+
- **Core Packages**: `transformers>=4.38.0`, `tiktoken>=0.6.0`, `regex>=2023.12.25`, `pandas>=2.0.0`, `numpy>=1.24.0`
