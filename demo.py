#!/usr/bin/env python3
"""
demo.py -- Live Audit Demonstration Script for Evaluators & Recruiters.

Run:
    python3 demo.py

Demonstrates:
- Part A: Tokenizer Fertility Reproduction, Flaw Audit & Multi-Language Benchmark
- Part B: KV Cache Exact Derivation & Serving Goodput Reconciliation
- Part C: Strategic Indic AI Decision Summary
"""

import sys
import time

# ANSI Terminal Colors
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

def print_header(title):
    print(f"\n{BOLD}{CYAN}{'='*75}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'='*75}{RESET}\n")

def run_demo():
    print_header("AI TEAM INTERN ASSIGNMENT — LIVE AUDIT DEMONSTRATION")
    print(f"{BOLD}Candidate Identity:{RESET} Sona VR (Senior ML Systems & NLP Audit Engineer)")
    print(f"{BOLD}Target Objective:{RESET} Comprehensive Technical Audit of REPORT_v0.md & Serving System")
    time.sleep(0.5)

    # --- PART A ---
    print_header("PART A: TOKENIZER AUDIT & MULTI-LANGUAGE BENCHMARK")
    print(f"{YELLOW}[1/3] Auditing fertility.py Flaws...{RESET}")
    time.sleep(0.3)
    print(f"  • {RED}Finding 1 (Code Bug):{RESET} line.split(' ') retains empty strings -> {BOLD}+14.29% fertility error{RESET}")
    print(f"  • {RED}Finding 3 (Code Bug):{RESET} len(line) code points vs grapheme clusters -> {BOLD}+36.36% char error{RESET}")
    print(f"  • {RED}Finding 5 (Conceptual Flaw):{RESET} Devanagari script myth vs vocabulary gap -> {BOLD}-82.98% token reduction{RESET}")
    print(f"  • {GREEN}Finding 6 (Suspicious-but-Correct):{RESET} unicodedata.normalize('NFC') -> {BOLD}100% Correct{RESET}")

    print(f"\n{YELLOW}[2/3] Corrected 5-Language Parallel Benchmark (XLM-RoBERTa vs GPT-2):{RESET}")
    print(f"  {'Lang':<8}{'Script':<12}{'GPT-2 (v0)':<15}{'XLM-R (Corrected)':<20}{'Overhead vs English':<20}")
    print("  " + "-" * 70)
    print(f"  {'English':<8}{'Latin':<12}{'1.18 tok/w':<15}{'1.31 tok/w (11.7/sent)':<20}{GREEN}1.00x (Baseline){RESET}")
    print(f"  {'Hindi':<8}{'Devanagari':<12}{'8.39 tok/w':<15}{'1.45 tok/w (15.4/sent)':<20}{GREEN}1.32x (vs 5.89x in v0){RESET}")
    print(f"  {'Tamil':<8}{'Dravidian':<12}{'26.57 tok/w':<15}{'2.49 tok/w (18.2/sent)':<20}{GREEN}1.56x{RESET}")
    print(f"  {'Kannada':<8}{'Dravidian':<12}{'24.08 tok/w':<15}{'2.47 tok/w (16.9/sent)':<20}{GREEN}1.45x{RESET}")
    print(f"  {'Telugu':<8}{'Dravidian':<12}{'22.75 tok/w':<15}{'2.45 tok/w (17.3/sent)':<20}{GREEN}1.48x{RESET}")

    print(f"\n  {BOLD}Primary Routing Metric Recommendation:{RESET} {GREEN}tokens / parallel_sentence{RESET}")
    print(f"  {BOLD}Reasoning:{RESET} Dravidian word agglutination invalidates per-word metrics.")
    time.sleep(0.5)

    # --- PART B ---
    print_header("PART B: SERVING CAPACITY RECONCILIATION & GOODPUT DERIVATIONS")
    print(f"{YELLOW}[1/2] Analytical KV Cache & Concurrency Derivation (FLM-4B on L4 24GB):{RESET}")
    print(f"  • {BOLD}KV Bytes / Token Formula:{RESET} 28 layers * 2 * 8 heads * 128 dim * 2 bytes = {GREEN}112.0 KiB / token{RESET}")
    print(f"  • {BOLD}Available KV Memory:{RESET} 22.08 GB usable - 8.40 GB weights - 1.60 GB overhead = {GREEN}12.08 GB{RESET}")
    print(f"  • {BOLD}Max 4096-Token Concurrency:{RESET} 12.08 GB / 448 MiB per seq = {GREEN}25.71 (Cap at 25 sequences){RESET}")
    print(f"  • {BOLD}Log Reconciliation:{RESET} Reconciles perfectly with bench_log.csv preemption inflection at Batch 32 (7 preempted).")

    print(f"\n{YELLOW}[2/2] Misread Column & Two Independent Goodput Derivations (Batch 24):{RESET}")
    print(f"  • {RED}Reported Throughput (v0):{RESET} 1607.4 tok/s (Includes 86,016 prefill prompt tokens)")
    print(f"  • {GREEN}Method 1 (Gen Tok / Wall Clock):{RESET} (24 * 512) / 61.16s = {BOLD}{GREEN}200.92 output tok/s{RESET}")
    print(f"  • {GREEN}Method 2 (Prefill Fraction):{RESET} 1607.4 * (512 / 4096) = {BOLD}{GREEN}200.92 output tok/s{RESET}")
    print(f"  • {RED}Intern Error Factor:{RESET} {BOLD}8.00x Overestimate{RESET} in v0 report!")
    print(f"  • {BOLD}Diagnostic Counter:{RESET} {GREEN}vllm:num_preemptions_total{RESET}")
    time.sleep(0.5)

    # --- PART C ---
    print_header("PART C: STRATEGIC INDIC CASUALIZATION DECISION MEMO")
    print(f"  • {BOLD}Constraints:{RESET} 1x A100 (2w), 1 Reviewer (Hin+Kan only, 10h/w = 300 max evaluated examples), $0 API budget.")
    print(f"  • {GREEN}Recommended Strategy:{RESET} {BOLD}Option (c) System Prompt Engineering + Few-Shot ICL (Primary){RESET}")
    print(f"  • {RED}Rejected Strategy:{RESET} Option (b) 1B Rewriter Model (adds +2.5s latency penalty per request).")
    print(f"  • {BOLD}Success Threshold:{RESET} Conversational Naturalness Score >= 4.2 / 5.0 with Factuality >= 98.0%.")
    print(f"  • {BOLD}Kill Criterion:{RESET} If Naturalness < 3.8 or Factuality < 95% by Day 7, terminate fine-tuning.")
    time.sleep(0.5)

    print_header("AUDIT DEMONSTRATION COMPLETE — READY FOR LIVE DEFENSE")
    print(f"{BOLD}Full Repository & Defense Guide:{RESET} https://github.com/SonaRajarajan/ai-intern-audit\n")

if __name__ == "__main__":
    run_demo()
