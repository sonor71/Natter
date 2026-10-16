"""Transparent training statistics; no false claim that tokens seen are unique."""
from __future__ import annotations

import math


def effective_tokens_per_step(batch_size: int, accumulation: int, context_size: int) -> int:
    if min(batch_size, accumulation, context_size) <= 0:
        raise ValueError("batch_size, accumulation and context_size must be positive")
    return batch_size * accumulation * context_size


def training_progress(current_step: int, initial_step: int, tokens_per_step: int,
                      corpus_tokens: int, val_loss: float) -> dict:
    """Estimate processed token positions using a constant batch/context configuration.

    Token positions are sampled with replacement; repeats are counted every time.
    The number is not unique training tokens and not a measure of intelligence.
    """
    if current_step < initial_step or tokens_per_step <= 0 or corpus_tokens <= 0:
        raise ValueError("Invalid step or corpus parameters")
    run_tokens = (current_step - initial_step) * tokens_per_step
    return {
        "run_tokens_seen": run_tokens,
        "sampled_corpus_passes_this_run": run_tokens / corpus_tokens,
        "val_perplexity": math.exp(val_loss) if math.isfinite(val_loss) and val_loss < 700 else float("inf"),
    }
