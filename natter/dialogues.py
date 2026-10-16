"""Обучение на одобренных ответах: prompt не включается в loss."""
from __future__ import annotations
import json
import random

import torch

from .settings import ROOT


def load_dialogues():
    return load_jsonl_pairs(ROOT / 'data' / 'dialogues' / 'approved.jsonl')


def load_jsonl_pairs(path, min_examples=4):
    if not path.exists():
        raise RuntimeError(f'Нет файла диалогов: {path}')
    examples = []
    for n, row in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not row.strip():
            continue
        record = json.loads(row)
        question, answer = record.get('user', '').strip(), record.get('assistant', '').strip()
        if question and answer:
            examples.append((question, answer))
    examples = list(dict.fromkeys(examples))
    if len(examples) < min_examples:
        raise RuntimeError(f'Недостаточно диалогов в {path}: {len(examples)} < {min_examples}')
    return examples


def split_dialogues(examples, seed):
    examples = list(examples)
    random.Random(seed).shuffle(examples)
    val_count = max(1, int(len(examples) * 0.15))
    return examples[val_count:], examples[:val_count]


def encode_dialogues(examples, tok, limit):
    data = []
    skipped = 0
    for user, answer in examples:
        prefix = [tok.bos_id] + tok.encode(f'ПОЛЬЗОВАТЕЛЬ: {user}\nНАТТЕР: ')
        suffix = tok.encode(answer) + [tok.eos_id]
        sequence = prefix + suffix
        if len(sequence) - 1 > limit:
            skipped += 1
            continue
        x = sequence[:-1]
        y = sequence[1:]
        # target[0 .. len(prefix)-2] = символы запроса, их не учим генерировать
        for i in range(len(prefix) - 1):
            y[i] = -100
        data.append((x, y))
    return data, skipped


def sample_batch(data, batch_size, pad_id, device, rng=None):
    items = (rng or random).choices(data, k=batch_size)
    longest = max(len(x) for x, _ in items)
    xs = [x + [pad_id] * (longest-len(x)) for x, _ in items]
    ys = [y + [-100] * (longest-len(y)) for _, y in items]
    return torch.tensor(xs, dtype=torch.long, device=device), torch.tensor(ys, dtype=torch.long, device=device)
