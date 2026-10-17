"""Explicit multi-turn dialogue representation, separate from single-turn SFT.

Each answer predicts its own tokens and EOS; user text provides context but
has loss=-100. At inference the SAME token representation is used, with
previous answers explicitly included in the prompt.

This enables context *conditioning* once a model is trained on it. It does
not by itself make the model able to reason about the context.
"""
from __future__ import annotations

import json
from pathlib import Path

from .dialogues import sample_batch

USER = "ПОЛЬЗОВАТЕЛЬ: "
ASSISTANT = "\nНАТТЕР: "


def validate_turns(turns):
    if not isinstance(turns, list) or len(turns) < 2:
        raise ValueError("Диалог должен содержать не менее 2 обменов")
    for i, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError(f"Обмен {i} не объект")
        question, answer = turn.get("user"), turn.get("assistant")
        if (not isinstance(question, str) or not isinstance(answer, str)
                or not question.strip() or not answer.strip()):
            raise ValueError(f"Обмен {i} должен иметь непустые user и assistant")
        if len(question) > 1500 or len(answer) > 3000:
            raise ValueError(f"Обмен {i} слишком длинный")
    return turns


def load_conversations(path: Path, min_examples: int = 2):
    if not path.is_file():
        raise RuntimeError(f"Файл с диалогами не найден: {path}")
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                turns = validate_turns(item["turns"])
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            rows.append(turns)
    if len(rows) < min_examples:
        raise RuntimeError(f"Требуется не менее {min_examples} диалогов; найдено {len(rows)}")
    return rows


def encode_conversation(turns, tok, limit: int):
    validate_turns(turns)
    sequence = [tok.bos_id]
    labels = [-100]  # prediction of BOS is never trained
    for turn in turns:
        question = turn["user"].strip()
        answer = turn["assistant"].strip()
        question_ids = tok.encode(f"{USER}{question}{ASSISTANT}")
        answer_ids = tok.encode(answer) + [tok.eos_id]
        # For autoregressive shift, each next answer token corresponds to
        # one position of x just before that token.
        sequence.extend(question_ids)
        labels.extend([-100] * len(question_ids))
        sequence.extend(answer_ids)
        labels.extend(answer_ids)
    if len(sequence) - 1 > limit:
        return None

    # For x[t] -> target[t+1], labels[t+1] is answer token iff trainable.
    xs = sequence[:-1]
    ys = labels[1:]
    # All user spans remain masked, all assistant spans learned.
    if not any(y != -100 for y in ys):
        raise RuntimeError("No assistant tokens included in training example")
    return xs, ys


def encode_conversations(conversations, tok, limit):
    result, skipped = [], 0
    for turns in conversations:
        item = encode_conversation(turns, tok, limit)
        if item is None:
            skipped += 1
        else:
            result.append(item)
    return result, skipped


def build_context(tok, history, question, context_size: int, output_budget: int,
                  max_turns: int = 4):
    """Inference prefix with BOS + up to max_turns previous exchanges.

    Previous assistant answer is followed by the same EOS as in SFT.
    The caller must use this format only with a multi-turn-trained checkpoint.
    """
    if not question.strip():
        raise ValueError("Question cannot be blank")
    if output_budget <= 0 or context_size - output_budget < 12:
        raise ValueError("Not enough room for both context and answer")
    limit = context_size - output_budget
    request = tok.encode(f"{USER}{question.strip()}{ASSISTANT}")
    previous = []
    for q, a in history[-max_turns:]:
        previous.append(
            tok.encode(f"{USER}{q.strip()}{ASSISTANT}") +
            tok.encode(a.strip()) + [tok.eos_id]
        )
    while previous and 1 + sum(map(len, previous)) + len(request) > limit:
        previous.pop(0)
    ids = [tok.bos_id]
    for segment in previous:
        ids.extend(segment)
    ids.extend(request)
    if len(ids) > limit:
        ids = [ids[0]] + ids[-(limit - 1):]
    return ids
