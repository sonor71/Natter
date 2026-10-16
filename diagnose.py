"""Reproducible inference diagnostics for NATTER's *own* checkpoints.

Unlike chat.py, this tool also tests raw-text continuation. The base model was
pretrained on prose, not role-labelled dialogues. A good loss does not imply
chat competence.

Run from the repository root:
    .\\venv\\Scripts\\python.exe diagnose.py --stage base --seed 42 --tokens 80
    .\\venv\\Scripts\\python.exe diagnose.py --stage base --temperature 0.8 --save
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import torch

from chat import select_checkpoint
from natter.checkpoints import load_checkpoint, make_model, validate_checkpoint
from natter.settings import ROOT, config, device_name, sha256_file
from natter.tokenizer import NatterTokenizer


DEFAULT_PROMPTS = (
    "Философия — это",
    "Свобода человека зависит от",
    "ПОЛЬЗОВАТЕЛЬ: Что такое свобода?\nНАТТЕР: ",
)
TOKENIZER_SAMPLE = "Привет, Наттер! Философия Ницше и Батая."


def main():
    parser = argparse.ArgumentParser(description="Диагностика качества генерации NATTER CORE")
    parser.add_argument("--stage", choices=("base", "dialogue", "auto"), default="base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 означает greedy: повторяемый результат")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--prompt", action="append", help="Своя фраза; можно повторять")
    parser.add_argument("--save", action="store_true", help="Сохранить результаты локально в logs/")
    args = parser.parse_args()
    if not 1 <= args.tokens <= 1024:
        parser.error("--tokens должен быть от 1 до 1024")
    if args.temperature < 0 or args.top_k < 0:
        parser.error("--temperature и --top-k должны быть неотрицательны")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = device_name()
    tok = NatterTokenizer(ROOT / "artifacts" / "tokenizer.model")
    ck_path = select_checkpoint(args.stage)
    ck = load_checkpoint(ck_path, device)
    model, arch = make_model(config(), tok, device)
    validate_checkpoint(ck, arch, sha256_file(tok.model_file))
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()

    encoded_sample = tok.encode(TOKENIZER_SAMPLE)
    decoded_sample = tok.decode(encoded_sample)
    unk_id = tok.sp.unk_id()
    unknowns = sum(x == unk_id for x in encoded_sample)
    print(f"Устройство: {device}; checkpoint: {ck_path.name}; шаг: {ck.get('step')}")
    print(f"Параметры: {sum(p.numel() for p in model.parameters()):,}; "
          f"BPE: {tok.vocab_size}; контекст: {model.context_size}")
    print(f"Roundtrip BPE: {'OK' if decoded_sample == TOKENIZER_SAMPLE else 'ОТЛИЧАЕТСЯ'}; "
          f"<unk>: {unknowns}; декодировано: {decoded_sample!r}")
    if unknowns:
        print("ВНИМАНИЕ: токенизатор выдаёт <unk> на обычном русском тексте.")

    forbidden = [i for i in (tok.pad_id, tok.bos_id, unk_id) if i >= 0]
    prompts = args.prompt or DEFAULT_PROMPTS
    outputs = []
    for index, prompt in enumerate(prompts, start=1):
        prompt_ids = [tok.bos_id] + tok.encode(prompt)
        prompt_ids = [prompt_ids[0]] + prompt_ids[1:][-max(1, model.context_size - 1):]
        if len(prompt_ids) > model.context_size:
            raise RuntimeError("Ошибка ограничения длины контекста")
        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits, _ = model(x)
            next_logits = logits[0, -1].float()
            if not torch.isfinite(next_logits).all():
                raise RuntimeError("NaN/Inf в логитах модели. Проверь checkpoint и обучение.")
            top_vals, top_ids = torch.topk(next_logits.softmax(-1), k=min(5, tok.vocab_size))
            top = [
                {"piece": tok.sp.id_to_piece(int(t)), "prob": round(float(v), 4)}
                for v, t in zip(top_vals.tolist(), top_ids.tolist())
            ]
            # Reset to the same seed for each prompt so a rerun is comparable.
            torch.manual_seed(args.seed + index)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed + index)
            result = model.generate(
                x, max_new_tokens=args.tokens, temperature=args.temperature,
                top_k=args.top_k, eos_id=tok.eos_id,
                forbidden_token_ids=forbidden,
            )
        new_ids = result[0, len(prompt_ids):].tolist()
        generated = tok.decode(new_ids)
        print(f"\n[{index}] Промпт: {prompt!r}")
        print("Предсказания следующего токена:", top)
        print("Продолжение:", repr(generated))
        outputs.append({
            "prompt": prompt, "response": generated, "top5": top,
            "generated_tokens": len(new_ids),
        })

    print("\nВАЖНО: бессвязный текст при низком loss может означать недообучение;")
    print("диалоговый промпт также не соответствует энциклопедическому предобучению.")
    if args.save:
        path = ROOT / "logs" / f"diagnose_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "checkpoint": ck_path.name, "checkpoint_step": ck.get("step"),
            "tokenizer_sha256": sha256_file(tok.model_file),
            "seed": args.seed, "temperature": args.temperature,
            "top_k": args.top_k, "tokens": args.tokens,
            "roundtrip_ok": decoded_sample == TOKENIZER_SAMPLE,
            "unknown_token_count": unknowns, "outputs": outputs,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Локальный отчёт:", path)


if __name__ == "__main__":
    main()
