# Part C — Decision Memo: Indic Conversational Casualization

**To**: Product & Engineering Leadership  
**From**: Senior ML Systems & NLP Research Engineer  
**Date**: September 6, 2026  
**Subject**: Technical & Resource-Constrained Strategy for Casual Indic Assistant Register  

---

## 1. ASSUMPTIONS

1. **Target Languages**: Hindi (`hin`), Kannada (`kan`), Tamil (`tam`), Telugu (`tel`), Bengali (`ben`), Marathi (`mar`).
2. **Hardware Constraint**: 1× NVIDIA A100 (80GB VRAM) dedicated for 2 weeks (336 total GPU hours).
3. **Reviewer Constraint**: 1 native-speaker reviewer fluent in **Hindi and Kannada ONLY**, allocated for 10 hours/week across 3 weeks (30 total reviewer hours; 20 hours available prior to launch review).
4. **Budget Constraint**: $0 external API budget (no OpenAI/Claude API calls allowed for data generation).
5. **Base Model**: FLM-4B-Instruct dense LLM deployed on single GPU.
6. **Review Throughput**: Human evaluation of conversational register, safety, and factual correctness requires ~4 minutes per example = **15 examples / hour**.

---

## 2. BACK-OF-THE-ENVELOPE ARITHMETIC

- **Human Review Capacity**:
  - Total reviewer hours available before launch review (Weeks 1 & 2):  
    $$20 \text{ hours} \times 15 \text{ ex/hour} = \mathbf{300 \text{ total evaluated examples}}$$
  - Per-language capacity for reviewed languages: **150 examples for Hindi**, **150 examples for Kannada**.
  - **Zero Capacity** for Tamil, Telugu, Bengali, Marathi (Reviewer cannot evaluate these languages).

- **Synthetic Data & SFT Fine-Tuning Budget (If Option A is used)**:
  - Local synthetic pair generation using FLM-4B or open local LLM (Qwen-2.5-72B quantized) on A100: ~100 pairs/hour = 10,000 synthetic pairs generated in 100 A100 GPU hours.
  - LoRA fine-tuning of FLM-4B (4.2B parameters) on A100: ~2 hours per 1,000 examples $\times$ 6 languages = 24 GPU hours per iteration.
  - Total GPU training hours needed for 6 languages: ~150 A100 hours (well within the 336-hour limit).
  - **Critical Bottleneck**: Synthetic data for 4 of the 6 target languages (Tamil, Telugu, Bengali, Marathi) **cannot be human-reviewed** before launch!

- **Serving Latency Cost (If Option B Rewriter is used)**:
  - Adding a <=1B rewriter model requires sequential 2-stage inference.
  - 1B rewriter decode step: ~25ms per token $\times$ 100 output tokens = **+2,500ms added end-to-end latency** per request, doubling TTFT and breaking serving SLAs.

---

## 3. SUCCESS METRIC

> [!IMPORTANT]
> **Primary Launch Metric**: **Conversational Naturalness Score $\ge 4.2 / 5.0$** while maintaining **Factual Accuracy Retention $\ge 98.0\%$** and **Safety/Toxicity Violation Rate $\le 0.5\%$**.
> 
> *Measurement*: Evaluated on a blinded 100-prompt test set by the native reviewer for Hindi and Kannada, comparing baseline FLM-4B against candidate outputs.

---

## 4. KILL CRITERION

> [!CAUTION]
> **Project Termination Trigger**:
> 
> **If by Day 7 (End of Week 1), candidate outputs fail to achieve Naturalness $\ge 3.8 / 5.0$ or drop Factual Accuracy below $95.0\%$ after 10 hours of native reviewer evaluation, IMMEDIATELY KILL the fine-tuning / rewriter track and default to standard system prompts.**

---

## 5. DAY-1 EXPERIMENT

On **Day 1**, execute a zero-cost, high-information System Prompt & In-Context Learning (ICL) benchmark:
1. Construct 4 system prompt register variants (Formal, Standard Casual, Localized Conversational, Few-Shot In-Context Pairs).
2. Generate outputs on FLM-4B-Instruct for 25 representative benchmark prompts in Hindi and Kannada.
3. Submit 50 generated pairs to the native reviewer for a 3-hour initial evaluation on Day 2 morning.
4. *Goal*: Determine whether System Prompt Engineering alone reaches Naturalness $\ge 4.0/5.0$ without any model fine-tuning.

---

## 6. RECOMMENDATION

> [!TIP]
> **Recommended Strategy: Staged Hybrid Approach — Primary Path: Option (c) System Prompt Engineering + Few-Shot In-Context Learning; Secondary Fallback: Option (a) Targeted LoRA SFT for Hindi & Kannada ONLY.**

### Justification & Trade-Off Analysis

1. **Why NOT Option (b) (1B Rewriter Model)**:
   - Option (b) adds 2.5 seconds of latency per request, doubles GPU memory usage during inference, and requires training a secondary model across 6 languages without reviewer validation for 4 languages.

2. **Why Option (c) (Prompt Engineering) is the Primary Path**:
   - **Zero Latency Overhead**: Adds no serving compute or memory cost.
   - **Covers All 6 Languages Immediately**: System prompts function uniformly across all languages supported by FLM-4B.
   - **Zero Training Cost**: Preserves 100% of A100 GPU hours.

3. **Why Option (a) (SFT) is Constrained to Hindi & Kannada Only (If Needed)**:
   - Unreviewed synthetic SFT data in languages without human reviewers (Tamil, Telugu, Bengali, Marathi) risks catastrophic hallucination, vulgar register regressions, or grammar degradation in production.
   - If Day 1 prompt engineering falls short for Hindi/Kannada, use 40 A100 hours to fine-tune lightweight LoRA adapters using a small 500-example high-precision synthetic dataset validated by the native reviewer.

---

## Comparative Matrix

| Approach | GPU Training Hours | Reviewer Hours Needed | Added Serving Latency | Risk in Unreviewed Languages | Recommendation Status |
|---|---|---|---|---|---|
| **(c) Prompt Engineering** | **0 hours** | **5 hours** | **0 ms** | **Low** | **PRIMARY LAUNCH PATH** |
| **(a) Targeted LoRA SFT** | **40 hours** | **15 hours** | **0 ms** | High (Hindi/Kannada only) | **FALLBACK (Hin/Kan)** |
| **(b) 1B Rewriter Model** | 120 hours | 30+ hours | +2500 ms | Severe | **REJECTED** |
