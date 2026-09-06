# Multilingual Evaluation Corpus (A1)

## Dataset Provenance & Overview

- **Source**: FLORES-200 parallel translation benchmark.
- **Languages Included**:
  1. `eng` — English (Latin script)
  2. `hin` — Hindi (Devanagari script)
  3. `tam` — Tamil (Dravidian, Tamil script)
  4. `kan` — Kannada (Dravidian, Kannada script)
  5. `tel` — Telugu (Dravidian, Telugu script)
- **Format**: Parallel aligned sentence files (`eng.txt`, `hin.txt`, `tam.txt`, `kan.txt`, `tel.txt`).
- **Sentence Count**: 25 parallel sentences per language (125 total sentences across 5 languages).
- **Preprocessing & Normalization**: All sentences pass through explicit Unicode NFC (Canonical Composition) normalization.

## Summary Statistics

| Language | Sentences | Whitespace Words | Grapheme Clusters | UTF-8 Bytes | Bytes / Word | Graphemes / Word |
|---|---|---|---|---|---|---|
| English (`eng`) | 25 | 223 | 1404 | 1404 | 6.30 | 6.30 |
| Hindi (`hin`) | 25 | 266 | 887 | 3719 | 13.98 | 3.33 |
| Tamil (`tam`) | 25 | 183 | 1107 | 4863 | 26.57 | 6.05 |
| Kannada (`kan`) | 25 | 171 | 971 | 4197 | 24.54 | 5.68 |
| Telugu (`tel`) | 25 | 177 | 799 | 4035 | 22.80 | 4.51 |

---

## Corpus Limitations & Real-World Caveats

> [!WARNING]
> **What This Evaluation Corpus Cannot Tell Us About Production Traffic:**
>
> 1. **Domain Mismatch & Translationese**: This evaluation benchmark consists of formal parallel news/domain sentences translated from English. Real production traffic to LLM endpoints consists of user queries, conversational dialog, slang, domain-specific code/JSON, and instructions. Parallel translated corpora often exhibit "translationese"—unnatural sentence structures that mimic English word order.
> 2. **Code-Switching & Script Mixing**: In production Indic AI systems (e.g., in India), users frequently engage in Hinglish, Tanglish, or script-mixing (e.g., writing Hindi or Tamil using ASCII Latin letters, or mixing English technical terms into Devanagari sentences like "GPU cluster memory crash ho gaya"). A purely monolingual parallel benchmark fails to capture tokenization efficiency on code-switched text.
> 3. **Morphological Typology Disparity**: Tamil, Kannada, and Telugu are highly agglutinative languages where case markers, tenses, and postpositions are fused into single long words. English and Hindi are isolating/inflected where postpositions/prepositions are separate whitespace-delimited words. Consequently, comparing `tokens / word` across languages misrepresents linguistic efficiency—a single Tamil word carries significantly more semantic payload than an English word.
> 4. **Sample Size & Variance**: While 25 parallel sentences provide an exact, reproducible evaluation for benchmarking tokenizer behavior, real production traffic spans millions of requests with wide length variance (from 2-word queries to 4000-token document summaries). Tokenizer overhead from special tokens or sentence-initial capitalization diminishes as sequence length increases.
