"""Fine-tune NATTER's OWN checkpoint on Russian supervised Q/A pairs.

The pretrained weights are never downloaded from a model hub.
Datasets:
  approved: user-curated data/dialogues/approved.jsonl (unchanged workflow)
  tiny-ru: CC0 filtered public split prepared by import_dialogues.py
"""
from __future__ import annotations

import argparse
import hashlib
import random
import time

import torch

from natter.checkpoints import load_checkpoint, make_model, save_checkpoint, validate_checkpoint
from natter.corpus import load_manifest
from natter.dialogues import load_dialogues, load_jsonl_pairs, split_dialogues, encode_dialogues, sample_batch
from natter.multiturn import load_conversations, encode_conversations
from natter.settings import ROOT, config, device_name, sha256_file
from natter.tokenizer import NatterTokenizer


@torch.no_grad()
def evaluate(model, data, tok, device, n=10, seed=1776):
    was_training = model.training
    model.eval()
    rng = random.Random(seed)
    losses = []
    for _ in range(n):
        x, y = sample_batch(data, 4, tok.pad_id, device, rng=rng)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == 'cuda')):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train(was_training)
    return sum(losses) / len(losses)


def dataset_splits(name, seed):
    if name in ('multiturn', 'curriculum'):
        base = ROOT / 'data' / 'dialogues' / name
        train_file, val_file = base / 'train.jsonl', base / 'val.jsonl'
        train = load_conversations(train_file, min_examples=5)
        val = load_conversations(val_file, min_examples=2)
        digest = ''.join(sha256_file(p) for p in (train_file, val_file))
        return train, val, hashlib.sha256(digest.encode()).hexdigest()
    if name == 'approved':
        path = ROOT / 'data' / 'dialogues' / 'approved.jsonl'
        examples = load_dialogues()
        if len(examples) < 10:
            raise RuntimeError('Для надёжной настройки лучше хотя бы 10 проверенных диалогов.')
        train, val = split_dialogues(examples, seed)
        digest = sha256_file(path)
        return train, val, digest
    if name in ('tiny-ru', 'tiny-chat'):
        folder = 'tiny_ru' if name == 'tiny-ru' else 'tiny_chat'
        base = ROOT / 'data' / 'dialogues' / folder
        paths = (base / 'train.jsonl', base / 'val.jsonl')
        if not all(p.is_file() for p in paths):
            raise RuntimeError('Сначала импортируй CC0 диалоги: python import_dialogues.py')
        train = load_jsonl_pairs(paths[0], min_examples=10)
        val = load_jsonl_pairs(paths[1], min_examples=5)
        fingerprint = ''.join(sha256_file(p) for p in paths)
        return train, val, hashlib.sha256(fingerprint.encode()).hexdigest()
    raise ValueError('Unsupported dialogue dataset')


def main():
    p = argparse.ArgumentParser(description='Обучение своего NATTER CORE на разговорах')
    p.add_argument('--dataset', choices=('approved', 'tiny-ru', 'tiny-chat', 'multiturn', 'curriculum'), default='approved')
    p.add_argument('--steps', type=int, default=500, help='Дополнительные шаги')
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--patience', type=int, default=6,
                   help='Остановить после N последовательных оценок без улучшения')
    args = p.parse_args()
    if args.steps <= 0 or args.eval_every <= 0 or args.patience <= 0:
        p.error('--steps и --eval-every должны быть положительными')

    cfg = config()
    seed = cfg['training']['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = device_name()
    manifest = load_manifest()
    tok = NatterTokenizer(ROOT / 'artifacts' / 'tokenizer.model')
    if sha256_file(tok.model_file) != manifest['tokenizer_sha256']:
        raise RuntimeError('Токенизатор не совпадает с корпусом.')

    train_raw, val_raw, data_sha = dataset_splits(args.dataset, seed)
    encoder = encode_conversations if args.dataset in ('multiturn', 'curriculum') else encode_dialogues
    train, bad1 = encoder(train_raw, tok, cfg['model']['context_size'])
    val, bad2 = encoder(val_raw, tok, cfg['model']['context_size'])
    if not train or not val:
        raise RuntimeError('Пары слишком длинные или данные пусты.')
    print(f'Dataset {args.dataset}: train={len(train):,}, val={len(val):,}, '
          f'пропущено слишком длинных={bad1 + bad2}', flush=True)
    print('Обучение: только ответы; вопрос маскируется в loss.')
    if args.dataset in ('multiturn', 'curriculum') and len(train) < 100:
        print('ПРЕДУПРЕЖДЕНИЕ: менее 100 многотуровых примеров. Это проверка механики,'
              ' не достаточный корпус для появления многошагового мышления.')
    model, arch = make_model(cfg, tok, device)
    optim = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    suffix = {'tiny-ru': 'dialogue_tiny_ru',
              'tiny-chat': 'dialogue_tiny_chat',
              'approved': 'dialogue',
              'multiturn': 'dialogue_multiturn',
              'curriculum': 'dialogue_curriculum'}[args.dataset]
    last = ROOT / 'checkpoints' / (suffix + '_last.pt')
    best = ROOT / 'checkpoints' / (suffix + '_best.pt')
    step = 0
    best_val = float('inf')
    bad_evals = 0
    if last.exists():
        ck = load_checkpoint(last, device)
        validate_checkpoint(ck, arch, manifest['tokenizer_sha256'])
        if ck.get('dialogues_sha256') != data_sha:
            raise RuntimeError(
                'Диалоговый корпус изменился. Нельзя продолжать с прежним '
                'состоянием автоматически; сделай резервную копию и выбери '
                'новую версию обучения.'
            )
        model.load_state_dict(ck['model_state_dict'])
        optim.load_state_dict(ck['optimizer_state_dict'])
        if ck.get('scaler_state_dict'):
            scaler.load_state_dict(ck['scaler_state_dict'])
        best_val = float(ck.get('best_val', float('inf')))
        bad_evals = int(ck.get('bad_evals', 0))
        step = int(ck['step'])
        print(f'Продолжаю диалоговое обучение с шага {step}', flush=True)
    else:
        if args.dataset == 'multiturn' and (ROOT / 'checkpoints' / 'dialogue_tiny_chat_best.pt').is_file():
            base = ROOT / 'checkpoints' / 'dialogue_tiny_chat_best.pt'
        else:
            base = ROOT / 'checkpoints' / 'pretrain_best.pt'
        if not base.is_file():
            raise RuntimeError('Нет собственных pretrained весов: сначала нужен train.py.')
        ck = load_checkpoint(base, device)
        validate_checkpoint(ck, arch, manifest['tokenizer_sha256'])
        model.load_state_dict(ck['model_state_dict'])
        print(f'Стартую от своих весов: {base.name}, шаг {ck.get("step")}', flush=True)

    start_time = time.time()
    until = step + args.steps

    def save(path):
        save_checkpoint(path, {
            'stage': 'dialogue',
            'dataset': args.dataset, 'step': step,
            'arch': arch, 'tokenizer_sha256': manifest['tokenizer_sha256'],
            'dialogues_sha256': data_sha,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val': best_val,
            'bad_evals': bad_evals,
        })

    try:
        while step < until:
            model.train()
            x, y = sample_batch(train, min(8, len(train)), tok.pad_id, device)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == 'cuda')):
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            step += 1
            if step % args.eval_every == 0 or step == until or step == 1:
                val_loss = evaluate(model, val, tok, device)
                print(f'Dialogue step {step}: train {loss.item():.4f} | '
                      f'val {val_loss:.4f} | {time.time()-start_time:.1f}s', flush=True)
                if val_loss < best_val:
                    best_val = val_loss
                    bad_evals = 0
                    save(best)
                    print('  Новый лучший диалоговый checkpoint:', best, flush=True)
                else:
                    bad_evals += 1
                save(last)
                if bad_evals >= args.patience:
                    print('Ранняя остановка: проверочная ошибка не улучшалась',
                          f'{args.patience} оценок.', flush=True)
                    break
    except KeyboardInterrupt:
        print('Остановка пользователем; сохраняю checkpoint.')
    finally:
        save(last)
        print('Диалоговая модель:', best if best.exists() else last)


if __name__ == '__main__':
    main()
