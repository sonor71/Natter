"""Logit filtering for a small locally trained causal language model.

These algorithms change decoding, not model weights or linguistic competence.
"""
from __future__ import annotations

import torch


def penalize_repeated_tokens(logits: torch.Tensor, generated: torch.Tensor, penalty: float) -> torch.Tensor:
    """HuggingFace-style sign-aware repetition penalty applied only to generated tokens.

    Repeated tokens are penalized once per unique token, regardless of frequency.
    The prompt is deliberately excluded.
    """
    if penalty < 1.0:
        raise ValueError("repetition_penalty must be >= 1.0")
    if penalty == 1.0 or generated.shape[-1] == 0:
        return logits
    for batch_index in range(generated.shape[0]):
        ids = generated[batch_index].unique()
        old = logits[batch_index, ids]
        logits[batch_index, ids] = torch.where(old < 0, old * penalty, old / penalty)
    return logits


def block_repeated_ngrams(logits: torch.Tensor, generated: torch.Tensor, size: int) -> torch.Tensor:
    """Forbid forming a previously generated n-gram for each batch item.

    The prompt is excluded: it can legitimately contain phrases the model
    should repeat in the answer.
    """
    if size == 0 or generated.shape[-1] < size - 1:
        return logits
    if size < 2:
        raise ValueError("no_repeat_ngram_size must be 0 or >= 2")
    for row in range(generated.shape[0]):
        tokens = generated[row].tolist()
        suffix = tuple(tokens[-(size - 1):])
        forbidden = {
            tokens[i + size - 1]
            for i in range(len(tokens) - size + 1)
            if tuple(tokens[i:i + size - 1]) == suffix
        }
        if forbidden:
            logits[row, list(forbidden)] = -float("inf")
    return logits


def nucleus_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Top-p nucleus sampling. Keep at least the single best token."""
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be within (0, 1]")
    if top_p == 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    remove = cumsum - sorted_probs >= top_p
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
    result = torch.full_like(logits, -float("inf"))
    result.scatter_(-1, sorted_indices, sorted_logits)
    return result
