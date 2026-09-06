#!/usr/bin/env python3
"""
compute_metrics.py -- Helper module for NLP string & metric computations.

Provides robust functions for:
- Unicode NFC normalization
- Grapheme cluster count (using regex \X or unicodedata boundary tracking)
- Whitespace word segmentation
- UTF-8 byte counting
- Ratio calculation (ratio of totals vs average of ratios)
"""

import regex
import unicodedata
from typing import List, Dict, Tuple, Any

def normalize_text(text: str, form: str = "NFC") -> str:
    """Normalize text using Unicode normalization (default NFC)."""
    return unicodedata.normalize(form, text.strip())

def count_grapheme_clusters(text: str) -> int:
    """Count user-perceived grapheme clusters in text using regex \\X."""
    # \\X matches a Unicode extended grapheme cluster
    return len(regex.findall(r'\X', text))

def count_code_points(text: str) -> int:
    """Count Python Unicode code points (len(text))."""
    return len(text)

def count_whitespace_words(text: str) -> int:
    """Count whitespace-separated words using .split() (collapsing multiple spaces)."""
    return len(text.split())

def count_naive_spaces(text: str) -> int:
    """Count words using line.split(' ') (flawed intern approach)."""
    return len(text.split(" "))

def count_utf8_bytes(text: str) -> int:
    """Count UTF-8 bytes for encoded text."""
    return len(text.encode("utf-8"))

def compute_corpus_stats(lines: List[str]) -> Dict[str, int]:
    """Compute aggregate counts for a list of normalized lines."""
    total_lines = len(lines)
    total_code_points = sum(count_code_points(l) for l in lines)
    total_graphemes = sum(count_grapheme_clusters(l) for l in lines)
    total_words_clean = sum(count_whitespace_words(l) for l in lines)
    total_words_naive = sum(count_naive_spaces(l) for l in lines)
    total_bytes = sum(count_utf8_bytes(l) for l in lines)

    return {
        "lines": total_lines,
        "code_points": total_code_points,
        "graphemes": total_graphemes,
        "words_clean": total_words_clean,
        "words_naive": total_words_naive,
        "bytes": total_bytes,
    }
