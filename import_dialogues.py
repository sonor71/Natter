"""Download (optional) and filter openly licensed Russian Q/A for NATTER SFT.

Source: GlayAI/Tiny-Russian-Dialogs, CC0-1.0, synthetic single-turn Q/A.
This is TEXT, not pretrained weights. Use only as a limited conversational
bootstrap; the original dataset contains repeated answers and unnatural text.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from natter.settings import ROOT, sha256_file
from natter.dialogue_filter import conversational_pair

DATASET_URL = (
    "https://huggingface.co/datasets/GlayAI/Tiny-Russian-Dialogs/"
    "resolve/main/Tiny-Russian-Dialogs.csv"
)
DATASET_SHA256 = "15aea2ae3ceb98c23082731fda3e389e7bcd09c5fbcc9c50b728b4b7b20fce36"
SOURCE = "GlayAI/Tiny-Russian-Dialogs"
LICENSE = "CC0-1.0"
BASE = ROOT / "data" / "dialogues" / "tiny_ru"
CHAT_BASE = ROOT / "data" / "dialogues" / "tiny_chat"
RAW = ROOT / "downloads" / "Tiny-Russian-Dialogs.csv"

# A different assistant's identity, location or claimed experiences
# must not be copied into NATTER's self-description.
IDENTITY_PATTERNS = (
    "я живу в облаке", "я работаю в облаке", "как chatgpt",
    "меня создал openai", "я — chatgpt", "я искусственный интеллект",
    "я ии", "я программа", "мне тоже приятно с тобой общаться",
)


def normalize(s):
    return " ".join(s.split()).strip()


def answer_key(answer):
    return normalize(answer).casefold()


def group_val(answer, validation_percent=10):
    # All exact-repeat answers stay entirely in either train or validation.
    digest = hashlib.sha256(answer_key(answer).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 100 < validation_percent


def is_usable(q, a):
    if not (3 <= len(q) <= 180 and 3 <= len(a) <= 350):
        return False
    if any(marker in a.casefold() for marker in IDENTITY_PATTERNS):
        return False
    if re.search(r"https?://|\{\{|\[\[|\x00", q + a):
        return False
    return True


def select_pairs(rows, max_per_answer=3, seed=1337, validation_percent=10, mode="general"):
    """Reservoir sampling per distinct answer; avoids first-N template bias."""
    if max_per_answer < 1:
        raise ValueError("max_per_answer must be >= 1")
    if mode not in ("general", "conversation"):
        raise ValueError("mode must be general or conversation")
    rng = random.Random(seed)
    seen_questions = set()
    buckets = defaultdict(list)
    counts = Counter()
    stats = Counter()
    for row in rows:
        stats["rows"] += 1
        q, a = normalize(row.get("q", "")), normalize(row.get("a", ""))
        if not is_usable(q, a):
            stats["filtered"] += 1
            continue
        if mode == "conversation" and not conversational_pair(q, a):
            stats["off_topic"] += 1
            continue
        key_q = q.casefold()
        if key_q in seen_questions:
            stats["duplicate_questions"] += 1
            continue
        seen_questions.add(key_q)
        key = answer_key(a)
        counts[key] += 1
        bucket = buckets[key]
        if len(bucket) < max_per_answer:
            bucket.append((q, a))
        else:
            position = rng.randrange(counts[key])
            if position < max_per_answer:
                bucket[position] = (q, a)
    train, val = [], []
    for key, pairs in buckets.items():
        if group_val(key, validation_percent):
            val.extend(pairs)
        else:
            train.extend(pairs)
    rng.shuffle(train)
    rng.shuffle(val)
    if len(train) < 10 or len(val) < 5:
        raise ValueError("Not enough usable samples after filtering and splitting")
    stats["distinct_answer_groups"] = len(buckets)
    stats["train"] = len(train)
    stats["val"] = len(val)
    return train, val, dict(stats)


def fetch_csv(destination=RAW):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        print("CSV найден:", destination)
    else:
        print("Загружаю диалоговый текстовый CSV (не модель):", DATASET_URL)
        tmp = destination.with_suffix(".part")
        request = urllib.request.Request(
            DATASET_URL, headers={"User-Agent": "NatterCorpusResearch/1.0"}
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as out:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    out.write(block)
            tmp.replace(destination)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
    actual_sha = sha256_file(destination)
    if actual_sha != DATASET_SHA256:
        raise RuntimeError(
            "CSV SHA256 differs from verified dataset file. "
            f"Expected {DATASET_SHA256}, got {actual_sha}. "
            "Do not train until source identity has been checked."
        )
    return destination


def write_pairs(path, pairs):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for q, a in pairs:
            fh.write(json.dumps({"user": q, "assistant": a}, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Подготовить диалоговые тексты CC0 для NATTER")
    parser.add_argument("--file", type=Path, help="Уже скачанный CSV (без интернета)")
    parser.add_argument("--mode", choices=("conversation", "general"), default="conversation",
                        help="conversation removes maths/translation tasks; general is legacy")
    parser.add_argument("--max-per-answer", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dry-run", action="store_true", help="Не записывать тренировочные пары")
    args = parser.parse_args()
    limit = args.max_per_answer if args.max_per_answer is not None else (6 if args.mode == "conversation" else 3)
    if limit < 1:
        parser.error("--max-per-answer должен быть >= 1")
    source = args.file if args.file else fetch_csv()
    if not source.is_file():
        parser.error(f"Не найден CSV: {source}")
    if args.file:
        print("ВНИМАНИЕ: локальный CSV указан вручную; проверь лицензию и происхождение.")
    with source.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not {"q", "a"} <= set(reader.fieldnames or []):
            parser.error("CSV должен содержать колонки q и a")
        train, val, stats = select_pairs(
            reader, max_per_answer=limit, seed=args.seed, mode=args.mode
        )
    print(f"Прочитано: {stats['rows']:,}; отброшено: {stats.get('filtered',0):,}; "
          f"повторных вопросов: {stats.get('duplicate_questions',0):,}")
    print(f"Групп ответов: {stats['distinct_answer_groups']:,}; "
          f"после фильтра train: {len(train):,}, val: {len(val):,}; "
          f"тематически исключено: {stats.get('off_topic', 0):,}")
    print('ПРИМЕРЫ ОТОБРАННЫХ ПАР (оцени качество прежде чем обучать):')
    for question, answer in train[:5]:
        print(f'  Ты: {question}\n  NATTER (обучающий ответ): {answer}')
    if args.dry_run:
        print("Dry run: выходные файлы не изменены")
        return
    target = CHAT_BASE if args.mode == "conversation" else BASE
    write_pairs(target / "train.jsonl", train)
    write_pairs(target / "val.jsonl", val)
    (target / "SOURCE.json").write_text(json.dumps({
        "mode": args.mode,
        "max_per_answer": limit,
        "name": SOURCE, "license": LICENSE,
        "url": DATASET_URL, "sha256": sha256_file(source),
        "notes": "Synthetic single-turn QA; repeated answers capped, "
                 "answer-disjoint train/val; not for philosophy evaluation.",
        "stats": stats,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Готово:", target / "train.jsonl", "и", target / "val.jsonl")
    print("Не изменены: исходные data/dialogues/approved.jsonl, модель, checkpoint и BPE.")


if __name__ == "__main__":
    main()
