from __future__ import annotations
import argparse
import hashlib
import time

import torch

from natter.checkpoints import load_checkpoint, make_model, save_checkpoint, validate_checkpoint
from natter.corpus import load_manifest
from natter.dialogues import load_dialogues, split_dialogues, encode_dialogues, sample_batch
from natter.settings import ROOT, config, device_name, sha256_file
from natter.tokenizer import NatterTokenizer


@torch.no_grad()
def evaluate(model, data, tok, device, n=10):
    model.eval()
    losses = []
    for _ in range(n):
        x, y = sample_batch(data, 2, tok.pad_id, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    return sum(losses) / len(losses)


def main():
    p = argparse.ArgumentParser(description='Отдельная стадия: обучение диалогам на проверенных ответах')
    p.add_argument('--steps', type=int, default=200, help='Дополнительные шаги')
    args = p.parse_args()
    if args.steps <= 0:
        raise SystemExit('Шагов должно быть > 0')
    cfg = config()
    device = device_name()
    manifest = load_manifest()
    tok = NatterTokenizer(ROOT / 'artifacts' / 'tokenizer.model')
    if sha256_file(tok.model_file) != manifest['tokenizer_sha256']:
        raise RuntimeError('Токенизатор изменился после подготовки корпуса.')
    examples = load_dialogues()
    dialogues_sha = sha256_file(ROOT / 'data' / 'dialogues' / 'approved.jsonl')
    train, val = split_dialogues(examples, cfg['training']['seed'])
    train, bad1 = encode_dialogues(train, tok, cfg['model']['context_size'])
    val, bad2 = encode_dialogues(val, tok, cfg['model']['context_size'])
    if not train or not val:
        raise RuntimeError('Нет допустимых диалогов после токенизации. Добавь короткие диалоги или увеличь контекст.')
    print(f'Dialogues: train={len(train)}, val={len(val)}; пропущено слишком длинных={bad1 + bad2}')
    model, arch = make_model(cfg, tok, device)
    optim = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    last = ROOT / 'checkpoints' / 'dialogue_last.pt'
    best = ROOT / 'checkpoints' / 'dialogue_best.pt'
    step = 0
    best_val = float('inf')
    if last.exists():
        ck = load_checkpoint(last, device)
        validate_checkpoint(ck, arch, manifest['tokenizer_sha256'])
        model.load_state_dict(ck['model_state_dict'])
        optim.load_state_dict(ck['optimizer_state_dict'])
        step = int(ck['step'])
        if ck.get('dialogues_sha256') == dialogues_sha:
            best_val = float(ck.get('best_val', float('inf')))
        else:
            print('Новые исправления изменили обучающие данные: сбрасываю лучшую проверочную ошибку.')
        print('Продолжение диалогового обучения с шага', step)
    else:
        base = ROOT / 'checkpoints' / 'pretrain_best.pt'
        if not base.exists():
            raise RuntimeError('Сначала нужна предобученная НАМИ модель: python train.py --steps ...')
        ck = load_checkpoint(base, device)
        validate_checkpoint(ck, arch, manifest['tokenizer_sha256'])
        model.load_state_dict(ck['model_state_dict'])
        print('Загружены собственные веса базовой модели:', base)
    start = time.time()
    for iteration in range(args.steps):
        model.train()
        x, y = sample_batch(train, min(4, len(train)), tok.pad_id, device)
        _, loss = model(x, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        step += 1
        if step == 1 or step % 50 == 0 or iteration == args.steps - 1:
            val_loss = evaluate(model, val, tok, device)
            print(f'Диалоги: step={step}, train={loss.item():.4f}, val={val_loss:.4f}, {time.time()-start:.1f}s')
            payload = {
                'stage': 'dialogue', 'step': step, 'arch': arch,
                'tokenizer_sha256': manifest['tokenizer_sha256'],
                'model_state_dict': model.state_dict(), 'optimizer_state_dict': optim.state_dict(),
                'best_val': min(best_val, val_loss), 'dialogues_sha256': dialogues_sha,
            }
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(best, payload)
            save_checkpoint(last, payload)
    print('Готово. Диалоговые веса:', best)


if __name__ == '__main__':
    main()
