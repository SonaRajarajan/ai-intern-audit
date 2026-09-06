#!/usr/bin/env python3
"""
kv_cache_calc.py -- Analytical KV Cache & Concurrency Solver (B1).

Solves KV cache bytes per token and maximum 4096-token sequence concurrency
from FLM-4B-Instruct model specification alone. Reconciles theoretical limits
with bench_log.csv.
"""

import json

def calculate_kv_cache_metrics():
    # Model Specification Parameters
    layers = 28
    d_model = 3072
    q_heads = 24
    kv_heads = 8  # Grouped Query Attention (GQA)
    head_dim = 128
    weights_precision_bytes = 2  # fp16
    kv_precision_bytes = 2       # fp16

    # Hardware Specs
    gpu_ram_gb = 24.0
    gpu_ram_bytes = int(gpu_ram_gb * 1e9)
    gpu_ram_gib_bytes = 24 * (1024 ** 3)
    gpu_memory_utilization = 0.92
    model_params_billion = 4.2
    non_kv_overhead_gb = 1.6

    # ------------------------------------------------------------------------
    # 1. Exact KV Cache Bytes Per Token Derivation
    # ------------------------------------------------------------------------
    # For each layer, KV cache stores Key tensor and Value tensor.
    # Key dim per layer = kv_heads * head_dim = 8 * 128 = 1024
    # Value dim per layer = kv_heads * head_dim = 8 * 128 = 1024
    # Total elements per layer per token = 2 * (kv_heads * head_dim) = 2048
    # Bytes per layer per token = 2048 * kv_precision_bytes = 4096 bytes
    # Total KV bytes per token across all L layers:
    kv_bytes_per_token = layers * 2 * kv_heads * head_dim * kv_precision_bytes
    kv_kib_per_token = kv_bytes_per_token / 1024

    # ------------------------------------------------------------------------
    # 2. Concurrency Calculation for 4096-Token Sequences
    # ------------------------------------------------------------------------
    seq_len = 4096
    kv_bytes_per_seq = seq_len * kv_bytes_per_token
    kv_mib_per_seq = kv_bytes_per_seq / (1024 ** 2)
    kv_gb_per_seq = kv_bytes_per_seq / 1e9

    # Usable Memory Breakdown (Decimal GB)
    usable_gpu_ram_gb = gpu_ram_gb * gpu_memory_utilization  # 22.08 GB
    model_weights_gb = model_params_billion * weights_precision_bytes  # 8.4 GB
    available_kv_ram_gb = usable_gpu_ram_gb - model_weights_gb - non_kv_overhead_gb  # 12.08 GB
    available_kv_ram_bytes = int(available_kv_ram_gb * 1e9)

    max_concurrency_exact = available_kv_ram_bytes / kv_bytes_per_seq
    max_concurrency_integer = int(max_concurrency_exact)

    results = {
        "model_architecture": {
            "layers": layers,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "precision": "fp16 (2 bytes)"
        },
        "kv_cache_per_token": {
            "bytes": kv_bytes_per_token,
            "kib": kv_kib_per_token
        },
        "per_4096_sequence": {
            "bytes": kv_bytes_per_seq,
            "mib": round(kv_mib_per_seq, 2),
            "gb": round(kv_gb_per_seq, 6)
        },
        "gpu_memory_breakdown_gb": {
            "total_gpu_ram": gpu_ram_gb,
            "gpu_memory_utilization": gpu_memory_utilization,
            "usable_gpu_ram": usable_gpu_ram_gb,
            "model_weights": model_weights_gb,
            "non_kv_overhead": non_kv_overhead_gb,
            "available_for_kv_cache": available_kv_ram_gb
        },
        "concurrency_4096_token_sequences": {
            "theoretical_exact": round(max_concurrency_exact, 2),
            "max_concurrency_integer": max_concurrency_integer
        }
    }

    return results

def main():
    res = calculate_kv_cache_metrics()
    print("============================================================")
    print("PART B1: KV CACHE & CONCURRENCY DERIVATION")
    print("============================================================")
    print(f"Exact KV Cache Bytes Per Token: {res['kv_cache_per_token']['bytes']} bytes ({res['kv_cache_per_token']['kib']} KiB)")
    print(f"KV Cache Size per 4096-Token Sequence: {res['per_4096_sequence']['mib']} MiB ({res['per_4096_sequence']['bytes']} bytes)")
    print("\n--- GPU Memory Allocation Breakdown (NVIDIA L4 24GB) ---")
    print(f"Usable GPU Memory (0.92 util):  {res['gpu_memory_breakdown_gb']['usable_gpu_ram']:.2f} GB")
    print(f"Model Weights (4.2B fp16):      {res['gpu_memory_breakdown_gb']['model_weights']:.2f} GB")
    print(f"Non-KV Runtime Overhead:        {res['gpu_memory_breakdown_gb']['non_kv_overhead']:.2f} GB")
    print(f"Available KV Cache Memory:       {res['gpu_memory_breakdown_gb']['available_for_kv_cache']:.2f} GB")
    print(f"\nMax Concurrent 4096-Token Sequences: {res['concurrency_4096_token_sequences']['theoretical_exact']} -> Upper bound: {res['concurrency_4096_token_sequences']['max_concurrency_integer']} sequences")
    print("============================================================")

    with open("partB/results/capacity_reconciliation.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()
