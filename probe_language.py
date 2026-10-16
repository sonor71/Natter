"""Inspect NATTER's *language* ability using unseen or training documents.

No training, no model downloads, no changes to checkpoints or tokenizer.
Displays a real passage continuation alongside what the model generates.
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime

import torch
from torch.nn import functional as F

from chat import select_checkpoint
from natter.checkpoints import load_checkpoint, make_model, validate_checkpoint
from natter.corpus import load_manifest
from natter.settings import ROOT, config, device_name, sha256_file
from natter.tokenizer import NatterTokenizer


def evaluate_continuation(model, prompt_ids, continuation_ids, device):
    """Mean next-token cross entropy for reference continuation only."""
    if not continuation_ids:
        raise ValueError("Continuation cannot be empty")
    full = prompt_ids + continuation_ids
    if len(full) - 1 > model.context_size:
        raise ValueError("Reference window exceeds context")
    x = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    target = torch.tensor(continuation_ids, dtype=torch.long, device=device)
    with torch.inference_mode():
        logits, _ = model(x)
        predicted = logits[0, len(prompt_ids)-1:]
        return float(F.cross_entropy(predicted.float(), target).item())


def choose_passages(manifest, tokenizer, split, count, prompt_tokens, target_tokens, seed):
    """Sample documents reproducibly; never mix train and validation."""
    candidates = list(manifest["val_documents" if split == "val" else "train_documents"])
    random.Random(seed).shuffle(candidates)
    selected = []
    for relative in candidates:
        path = ROOT / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8-sig")
        ids = tokenizer.encode(text)
        # Prefix uses BOS; full reference must fit the model context.
        required = prompt_tokens + target_tokens
        if len(ids) < required:
            continue
        selected.append((relative, [tokenizer.bos_id] + ids[:required]))
        if len(selected) == count:
            break
    if len(selected) < count:
        raise RuntimeError(
            f"Only {len(selected)} suitable files in {split}; "
            "reduce --examples, --prefix-tokens or --tokens."
        )
    return selected


def main():
    p = argparse.ArgumentParser(description="NATTER: сравнение генерации с настоящими продолжениями статей")
    p.add_argument("--stage", choices=("base", "dialogue", "auto"), default="base")
    p.add_argument("--split", choices=("val", "train"), default="val")
    p.add_argument("--examples", type=int, default=3)
    p.add_argument("--prefix-tokens", type=int, default=64)
    p.add_argument("--tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", type=float, default=1.15)
    p.add_argument("--no-repeat-ngram", type=int, default=3)
    p.add_argument("--save", action="store_true")
    args = p.parse_args()

    if args.examples < 1 or args.prefix_tokens < 2 or args.tokens < 1:
        p.error("Нужны positive --examples, --tokens и --prefix-tokens >= 2")
    if not (0 < args.top_p <= 1) or args.top_k < 0 or args.repetition_penalty < 1:
        p.error("Некорректные параметры декодирования")
    if args.no_repeat_ngram not in (0,) and args.no_repeat_ngram < 2:
        p.error("--no-repeat-ngram должен быть 0 или >= 2")

    cfg = config()
    device = device_name()
    tokenizer = NatterTokenizer(ROOT / "artifacts" / "tokenizer.model")
    checkpoint_path = select_checkpoint(args.stage)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model, architecture = make_model(cfg, tokenizer, device)
    validate_checkpoint(checkpoint, architecture, sha256_file(tokenizer.model_file))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    if args.prefix_tokens + args.tokens > model.context_size:
        p.error("Сумма --prefix-tokens и --tokens не должна превышать контекст модели")

    manifest = load_manifest()
    if manifest["tokenizer_sha256"] != sha256_file(tokenizer.model_file):
        raise RuntimeError("Corpus manifest and tokenizer do not match")
    if checkpoint.get("corpus_sha256") != manifest.get("corpus_sha256"):
        print("ВНИМАНИЕ: checkpoint создан на другой версии корпуса; сравнение потерь ограничено.")

    cases = choose_passages(
        manifest, tokenizer, args.split, args.examples,
        args.prefix_tokens, args.tokens, args.seed
    )
    print(f"Модель: {checkpoint_path.name}, шаг {checkpoint.get('step')}")
    print(f"Документы: {args.split}; seed: {args.seed}")
    forbidden = [i for i in (tokenizer.pad_id, tokenizer.bos_id, tokenizer.sp.unk_id()) if i >= 0]
    output = []
    for n, (relative, all_ids) in enumerate(cases, 1):
        prefix = all_ids[:args.prefix_tokens + 1]
        reference = all_ids[len(prefix):]
        loss = evaluate_continuation(model, prefix, reference, device)
        torch.manual_seed(args.seed + n)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + n)
        x = torch.tensor([prefix], dtype=torch.long, device=device)
        generated = model.generate(
            x, max_new_tokens=len(reference), temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram,
            forbidden_token_ids=forbidden, eos_id=tokenizer.eos_id
        )
        actual = tokenizer.decode(generated[0, len(prefix):].tolist())
        expected = tokenizer.decode(reference)
        snippet = tokenizer.decode(prefix[1:])
        print(f"\n[{n}] {relative}")
        print(f"Reference continuation loss: {loss:.4f} (lower = better prediction)")
        print("Фрагмент статьи:", repr(snippet))
        print("Настоящее продолжение:", repr(expected))
        print("Продолжение Наттера:", repr(actual))
        output.append({
            "document": relative, "reference_loss": loss, "prefix": snippet,
            "reference": expected, "model_output": actual,
        })

    if args.save:
        path = ROOT / "logs" / f"language_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "step": checkpoint.get("step"), "checkpoint": checkpoint_path.name,
            "split": args.split, "seed": args.seed, "results": output,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\nОтчёт сохранён локально:", path)

    print("\nВажно: оценка предсказания статьи не доказывает навыки диалога или философии.")


if __name__ == "__main__":
    main()
