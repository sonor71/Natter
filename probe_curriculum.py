"""Evaluate NATTER's own curriculum SFT on *held-out scenarios*.

Reports generated responses, simple task-specific checks and reference-answer
token loss. No checkpoint modification and no claim of general reasoning.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

import torch
from torch.nn import functional as F

from chat import load_brain, select_checkpoint
from natter.curriculum import SCENARIOS
from natter.multiturn import build_context
from natter.settings import ROOT


def select_cases(rows, per_skill: int = 2, seed: int = 42):
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["skill"]].append(row)
    chosen = []
    for skill in sorted(grouped):
        pool = grouped[skill][:]
        rng.shuffle(pool)
        seen = set()
        for case in pool:
            if case["scenario"] not in seen:
                chosen.append(case)
                seen.add(case["scenario"])
            if len(seen) >= per_skill:
                break
    return chosen


def task_check(response: str, case: dict):
    """Heuristic checks; not semantic understanding or human evaluation."""
    skill = case["skill"]
    if skill == "philosophy":
        # Philosophical correctness requires human review, not string matching.
        return None
    _, a, b, c, name = case_values(case["scenario"])
    response = response.casefold()
    if skill == "chain":
        return b in response and c in response
    if skill == "memory":
        return name.casefold() in response
    if skill == "converse":
        return a in response and any(x in response for x in ("нельзя", "не следует", "нет", "невозможно"))
    if skill == "contradiction":
        return b in response and any(x in response for x in ("противореч", "несовмест"))
    if skill == "uncertainty":
        return b in response and any(x in response for x in ("неизвест", "нет", "нельзя", "не обоснован"))
    return False


def case_values(scenario_index: int):
    a, b, c, name = SCENARIOS[scenario_index]
    return scenario_index, a, b, c, name


@torch.no_grad()
def target_loss(model, prefix: list[int], answer_tokens: list[int], device: str):
    # All answer tokens are predicted, including EOS.
    sequence = prefix + answer_tokens
    if len(sequence) - 1 > model.context_size:
        raise ValueError("Scoring sequence exceeds model context")
    x = torch.tensor([sequence[:-1]], dtype=torch.long, device=device)
    logits, _ = model(x)
    offset = len(prefix) - 1
    predicted = logits[0, offset:offset + len(answer_tokens)].float()
    target = torch.tensor(answer_tokens, dtype=torch.long, device=device)
    return float(F.cross_entropy(predicted, target).item())


def main():
    p = argparse.ArgumentParser(description="Оценка нового нейросетевого рассуждения NATTER")
    p.add_argument("--per-skill", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.35)
    p.add_argument("--tokens", type=int, default=120)
    p.add_argument("--save", action="store_true")
    args = p.parse_args()
    if args.per_skill < 1 or args.tokens < 1:
        p.error("Параметры должны быть положительными")

    source = ROOT / "data" / "dialogues" / "curriculum" / "val.jsonl"
    if not source.is_file():
        raise SystemExit("Нет curriculum/val.jsonl. Сначала: python build_curriculum.py")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = select_cases(rows, per_skill=args.per_skill, seed=args.seed)

    checkpoint = select_checkpoint("curriculum")
    model, tok, device = load_brain("curriculum")
    print(f"Испытываем: {checkpoint.name}; проверочные сценарии, не вошедшие в train.")
    print("Оценка приблизительная; при плохом русском языке закономерны низкие результаты.")
    correct = defaultdict(int)
    counts = defaultdict(int)
    detailed = []

    for index, case in enumerate(cases):
        turns = case["turns"]
        question = turns[-1]["user"]
        expected = turns[-1]["assistant"]
        history = [(t["user"], t["assistant"]) for t in turns[:-1]]
        budget = min(args.tokens, model.context_size // 3)
        prefix = build_context(tok, history, question, model.context_size, budget)
        expected_ids = tok.encode(expected) + [tok.eos_id]
        loss = target_loss(model, prefix, expected_ids, device)
        torch.manual_seed(args.seed + index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + index)
        input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
        generated = model.generate(
            input_ids,
            max_new_tokens=budget, temperature=args.temperature,
            top_k=40, top_p=0.95,
            repetition_penalty=1.1, no_repeat_ngram_size=3,
            eos_id=tok.eos_id,
            forbidden_token_ids=[i for i in (tok.bos_id, tok.pad_id, tok.sp.unk_id()) if i >= 0],
        )
        output = tok.decode(generated[0, len(prefix):].tolist()).strip()
        match = task_check(output, case)
        if match is not None:
            counts[case["skill"]] += 1
            correct[case["skill"]] += int(match)
        detailed.append({
            "skill": case["skill"], "scenario": case["scenario"],
            "question": question, "expected": expected,
            "generated": output, "heuristic_match": match, "reference_loss": loss,
        })
        label = "MANUAL REVIEW" if match is None else ("PASS" if match else "FAIL")
        print(f"\n{case['skill']} | scenario {case['scenario']} | loss {loss:.3f} | check {label}")
        print("Вопрос:", question)
        print("Ожидаемый смысл:", expected)
        print("НАТТЕР:", output)
    passed = sum(correct.values())
    total = sum(counts.values())
    print(f"\nЭвристические проверки: {passed}/{total}; это НЕ тест общего интеллекта. Философские ответы оцениваются вручную.")
    for skill in sorted(counts):
        print(f"  {skill}: {correct[skill]}/{counts[skill]}")
    if args.save:
        from datetime import datetime
        report = ROOT / "logs" / f"curriculum_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps({
            "checkpoint": checkpoint.name, "seed": args.seed,
            "passed": passed, "total": total, "cases": detailed,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Локальный отчёт:", report)


if __name__ == "__main__":
    main()
