#!/usr/bin/env python3
"""
generate_plots.py -- Generate publication-quality figures for README documentation.

Generates:
1. partA/figures/tokenizer_fertility_comparison.png
2. partA/figures/tokens_per_sentence_parallel.png
3. partB/figures/throughput_anomaly_reconciliation.png
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import matplotlib.pyplot as plt
import numpy as np

def generate_partA_plots():
    os.makedirs("partA/figures", exist_ok=True)
    plt.style.use('ggplot')

    langs = ['English', 'Hindi', 'Tamil', 'Kannada', 'Telugu']
    gpt2_fert = [1.18, 8.39, 26.57, 24.08, 22.75]
    xlm_fert = [1.31, 1.45, 2.49, 2.47, 2.45]

    # Chart 1: Tokens Per Word Comparison
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(langs))
    width = 0.35

    rects1 = ax.bar(x - width/2, gpt2_fert, width, label='GPT-2 (v0 English-Centric)', color='#e74c3c')
    rects2 = ax.bar(x + width/2, xlm_fert, width, label='XLM-RoBERTa (Indic-Aware)', color='#2ecc71')

    ax.set_ylabel('Tokens Per Whitespace Word', fontsize=11, fontweight='bold')
    ax.set_title('Cross-Language Tokenizer Fertility (GPT-2 vs XLM-RoBERTa)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=10, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9, fontweight='bold')

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('partA/figures/tokenizer_fertility_comparison.png')
    plt.close()

    # Chart 2: Tokens Per Parallel Sentence (Routing Metric)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    gpt2_sent = [10.56, 89.32, 194.52, 164.72, 161.08]
    xlm_sent = [11.68, 15.40, 18.20, 16.92, 17.32]

    rects1 = ax.bar(x - width/2, gpt2_sent, width, label='GPT-2 (v0 Baseline)', color='#3498db')
    rects2 = ax.bar(x + width/2, xlm_sent, width, label='XLM-RoBERTa (Indic-Aware)', color='#2ecc71')

    ax.set_ylabel('Tokens Per Parallel Sentence', fontsize=11, fontweight='bold')
    ax.set_title('Operational Token Overhead Per Parallel Sentence Across Languages', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=10, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8, fontweight='bold')

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig('partA/figures/tokens_per_sentence_parallel.png')
    plt.close()
    print("Part A plots generated successfully!")

def generate_partB_plots():
    os.makedirs("partB/figures", exist_ok=True)
    plt.style.use('ggplot')

    batches = [4, 8, 16, 24, 32, 48]
    reported_tok_s = [565.4, 902.6, 1311.4, 1607.4, 1384.0, 1298.5]
    goodput_tok_s = [70.67, 112.84, 163.94, 200.92, 172.99, 162.31]
    preempted_seqs = [0, 0, 0, 0, 7, 23]

    fig, ax1 = plt.subplots(figsize=(10, 5.5), dpi=300)

    color1 = '#2980b9'
    color2 = '#27ae60'
    color3 = '#c0392b'

    ax1.set_xlabel('Batch Size (Long Prompt = 3584, Gen = 512)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Throughput (Tokens / Second)', color=color1, fontsize=11, fontweight='bold')
    
    line1 = ax1.plot(batches, reported_tok_s, marker='o', linewidth=2.5, color=color1, label='Reported Total Tok/s (Prefill + Gen)')
    line2 = ax1.plot(batches, goodput_tok_s, marker='s', linewidth=2.5, color=color2, label='True Generation Goodput Tok/s (Decode Only)')
    ax1.tick_params(axis='y', labelcolor=color1)

    # Highlight Preemption Inflection Point at Batch 24
    ax1.axvline(x=24, color='#f39c12', linestyle='--', linewidth=1.5, label='KV Capacity Cap (24 Seqs)')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Preempted Sequences (KV Cache Thrashing)', color=color3, fontsize=11, fontweight='bold')
    line3 = ax2.plot(batches, preempted_seqs, marker='^', linewidth=2.5, color=color3, linestyle=':', label='Preempted Sequences')
    ax2.tick_params(axis='y', labelcolor=color3)
    ax2.grid(False)

    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center left', fontsize=9)

    plt.title('Serving Throughput vs True Generation Goodput & Preemption Inflection', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('partB/figures/throughput_anomaly_reconciliation.png')
    plt.close()
    print("Part B plots generated successfully!")

def main():
    generate_partA_plots()
    generate_partB_plots()

if __name__ == "__main__":
    main()
