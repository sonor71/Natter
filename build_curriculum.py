"""Generate a local, reproducible reasoning curriculum (no API/model downloads).

Usage:
  python build_curriculum.py
  python build_curriculum.py --dry-run
  python build_curriculum.py --force   # ONLY after archiving old training run
"""
from __future__ import annotations

import argparse
from natter.curriculum import build, write_curriculum
from natter.settings import ROOT


def main():
    p = argparse.ArgumentParser(description="NATTER: собрать проверяемые многотуровые задания")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="Перезаписать отдельный локальный датасет curriculum")
    args = p.parse_args()
    examples = build(seed=args.seed)
    print(f"Обучающих диалогов: {len(examples['train'])}; "
          f"проверочных: {len(examples['val'])}")
    print("Проверочные сценарии не встречаются среди учебных.")
    print("Пример задания:")
    for turn in examples["train"][0]["turns"]:
        print("Пользователь:", turn["user"])
        print("Наттер:", turn["assistant"])
    if args.dry_run:
        print("Dry-run: никакие файлы не записаны.")
        return
    result = write_curriculum(ROOT, seed=args.seed, overwrite=args.force)
    print("Подготовлено: data/dialogues/curriculum/train.jsonl и val.jsonl")
    print("Метаданные:", result)
    print("Не изменены: предыдущие датасеты, checkpoint и BPE.")


if __name__ == "__main__":
    main()
