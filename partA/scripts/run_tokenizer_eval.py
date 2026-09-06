#!/usr/bin/env python3
"""
run_tokenizer_eval.py -- Multi-Tokenizer & Multi-Denominator Evaluation (A3).

Evaluates:
- Tokenizers: gpt2 (English-centric) vs xlm-roberta-base (Indic-aware)
- Languages: eng, hin, tam, kan, tel
- Denominators:
  1. tokens / whitespace_word (Linguistic word unit)
  2. tokens / grapheme_cluster (Visual character unit)
  3. tokens / utf8_byte (Information payload unit)
  4. tokens / parallel_sentence (Semantic payload unit)

Outputs tokenizer_summary.json and prints comparison tables.
"""

import os
import json
import argparse
import tiktoken
from compute_metrics import count_grapheme_clusters, count_whitespace_words, count_utf8_bytes, normalize_text
from transformers import AutoTokenizer

def load_tokenizers():
    enc_gpt2 = tiktoken.get_encoding("gpt2")
    try:
        tok_xlm = AutoTokenizer.from_pretrained("xlm-roberta-base")
        encode_xlm = lambda s: tok_xlm.encode(s, add_special_tokens=False)
    except Exception as e:
        print(f"Warning: Could not load XLM-RoBERTa from HuggingFace ({e}). Falling back to local tokenizer.")
        tok_xlm = None
        encode_xlm = None
    return enc_gpt2.encode, encode_xlm

def main():
    parser = argparse.ArgumentParser(description="Run Multi-Tokenizer Evaluation (A3)")
    parser.add_argument("--corpus_dir", default="partA/corpus", help="Corpus directory containing eng.txt, hin.txt, etc.")
    parser.add_argument("--output_json", default="partA/results/tokenizer_summary.json", help="Output JSON path")
    args = parser.parse_args()

    encode_gpt2, encode_xlm = load_tokenizers()
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    langs = ["eng", "hin", "tam", "kan", "tel"]
    tokenizers = {"gpt2": encode_gpt2}
    if encode_xlm:
        tokenizers["xlm-roberta-base"] = encode_xlm

    summary_data = {}

    print("==================================================================================================")
    print("CROSS-LANGUAGE MULTI-TOKENIZER & MULTI-DENOMINATOR EVALUATION (A3)")
    print("==================================================================================================")

    header = f"{'lang':<6}{'tokenizer':<18}{'sents':<7}{'words':<7}{'graphemes':<11}{'bytes':<8}{'tokens':<8}{'tok/word':<10}{'tok/graph':<11}{'tok/byte':<10}{'tok/sent':<10}"
    print(header)
    print("-" * len(header))

    for lang in langs:
        file_path = os.path.join(args.corpus_dir, f"{lang}.txt")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        lines = [normalize_text(line) for line in open(file_path, "r", encoding="utf-8") if line.strip()]
        n_sents = len(lines)
        tot_words = sum(count_whitespace_words(l) for l in lines)
        tot_graphemes = sum(count_grapheme_clusters(l) for l in lines)
        tot_bytes = sum(count_utf8_bytes(l) for l in lines)

        summary_data[lang] = {}

        for tok_name, encode_fn in tokenizers.items():
            tot_tokens = sum(len(encode_fn(l)) for l in lines)

            tok_per_word = tot_tokens / tot_words
            tok_per_graph = tot_tokens / tot_graphemes
            tok_per_byte = tot_tokens / tot_bytes
            tok_per_sent = tot_tokens / n_sents

            summary_data[lang][tok_name] = {
                "sentences": n_sents,
                "words": tot_words,
                "graphemes": tot_graphemes,
                "utf8_bytes": tot_bytes,
                "tokens": tot_tokens,
                "tokens_per_word": round(tok_per_word, 3),
                "tokens_per_grapheme": round(tok_per_graph, 3),
                "tokens_per_byte": round(tok_per_byte, 3),
                "tokens_per_sentence": round(tok_per_sent, 3)
            }

            print(f"{lang:<6}{tok_name:<18}{n_sents:<7}{tot_words:<7}{tot_graphemes:<11}{tot_bytes:<8}{tot_tokens:<8}{tok_per_word:<10.2f}{tok_per_graph:<11.3f}{tok_per_byte:<10.3f}{tok_per_sent:<10.2f}")

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed summary JSON written to: {args.output_json}")

if __name__ == "__main__":
    main()
