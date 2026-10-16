from __future__ import annotations
import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from natter.checkpoints import load_checkpoint, make_model, save_checkpoint, validate_checkpoint
from natter.corpus import StreamData, load_manifest
from natter.settings import ROOT, config, device_name, sha256_file
from natter.tokenizer import NatterTokenizer


def learning_rate(step, opts):
    peak = opts['learning_rate']
    low = opts['min_learning_rate']
    warmup = opts['warmup_steps']
    decay = opts['decay_steps']
    if warmup and step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(1, decay - warmup)))
    return low + 0.5 * (peak - low) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, train_stream, val_stream, cfg, device, train_context):
    was_training = model.training
    model.eval()
    result = {}
    for name, stream in [('train', train_stream), ('val', val_stream)]:
        losses = []
        for _ in range(cfg['training']['eval_batches']):
            xb, yb = stream.batch(cfg['training']['batch_size'], train_context, device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == 'cuda')):
                _, loss = model(xb, yb)
            losses.append(loss.item())
        result[name] = sum(losses) / len(losses)
    model.train(was_training)
    return result


def main():
    parser = argparse.ArgumentParser(description='NATTER CORE: базовое обучение языковой модели с нуля')
    parser.add_argument('--steps', type=int, default=500, help='Сколько ДОПОЛНИТЕЛЬНЫХ обновлений сделать')
    parser.add_argument('--allow-data-change', action='store_true', help='Продолжить на изменённом корпусе без смены BPE')
    args = parser.parse_args()
    if args.steps <= 0:
        raise SystemExit('--steps должен быть положительным')

    cfg = config()
    opts = cfg['training']
    random.seed(opts['seed'])
    np.random.seed(opts['seed'])
    torch.manual_seed(opts['seed'])
    device = device_name()
    if device == 'cuda':
        torch.cuda.manual_seed_all(opts['seed'])
    print('Device:', device, '| CUDA:', torch.cuda.is_available())
    if device == 'cpu':
        print('ВНИМАНИЕ: обучение на CPU будет медленным. На RTX 3080 нужна CUDA-сборка PyTorch.')

    manifest = load_manifest()
    tok = NatterTokenizer(ROOT / 'artifacts' / 'tokenizer.model')
    if sha256_file(tok.model_file) != manifest['tokenizer_sha256']:
        raise RuntimeError('Изменился токенизатор после prepare.py. Нужно подготовить корпус заново, затем проверить совместимость checkpoint.')
    model, arch = make_model(cfg, tok, device)
    print(f"Параметры: {sum(p.numel() for p in model.parameters()):,} | контекст: {arch['context_size']} токенов")
    train_data = StreamData('train')
    val_data = StreamData('val')
    train_context = min(arch['context_size'], len(train_data.tokens)-2, len(val_data.tokens)-2)
    if train_context < 128:
        raise RuntimeError('Корпус слишком мал: нужны дополнительные документы в data/raw/.')
    if train_context < arch['context_size']:
        print(f'ВНИМАНИЕ: для малого корпуса длина тренировочного окна временно {train_context} вместо {arch["context_size"]}.')
    optim = torch.optim.AdamW(model.parameters(), lr=opts['learning_rate'], weight_decay=opts['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    last_path = ROOT / 'checkpoints' / 'pretrain_last.pt'
    best_path = ROOT / 'checkpoints' / 'pretrain_best.pt'
    step = 0
    best_val = float('inf')
    bad_evals = 0
    if last_path.exists():
        ckpt = load_checkpoint(last_path, device)
        validate_checkpoint(ckpt, arch, manifest['tokenizer_sha256'])
        if ckpt.get('corpus_sha256') != manifest['corpus_sha256']:
            if not args.allow_data_change:
                raise RuntimeError('Данные изменились. Для продолжения с тем же токенизатором укажи --allow-data-change.')
            print('Корпус обновлён: сбрасываю best_val и счётчик остановки. Старый best.pt оставлен как архив.')
        else:
            best_val = ckpt.get('best_val', float('inf'))
            bad_evals = ckpt.get('bad_evals', 0)
        model.load_state_dict(ckpt['model_state_dict'])
        optim.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt.get('scaler_state_dict'):
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        step = int(ckpt['step'])
        print(f'Продолжаю обучение с шага {step}; текущая задача: ещё {args.steps} шагов.')
    else:
        print('Предыдущий checkpoint не найден: обучение новых весов с нуля.')
    until = step + args.steps
    start = time.time()
    best_is_current = False

    def save(path):
        save_checkpoint(path, {
            'stage': 'pretrain', 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(), 'scaler_state_dict': scaler.state_dict(),
            'step': step, 'best_val': best_val, 'bad_evals': bad_evals,
            'arch': arch, 'tokenizer_sha256': manifest['tokenizer_sha256'],
            'corpus_sha256': manifest['corpus_sha256'],
        })

    try:
        while step < until:
            model.train()
            optim.zero_grad(set_to_none=True)
            lr = learning_rate(step, opts)
            for group in optim.param_groups:
                group['lr'] = lr
            for _ in range(opts['grad_accum_steps']):
                xb, yb = train_data.batch(opts['batch_size'], train_context, device)
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == 'cuda')):
                    _, loss = model(xb, yb)
                scaler.scale(loss / opts['grad_accum_steps']).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
            scaler.step(optim)
            scaler.update()
            step += 1
            if step == 1 or step % opts['eval_every'] == 0 or step == until:
                scores = evaluate(model, train_data, val_data, cfg, device, train_context)
                print(f"step {step:6d} | train {scores['train']:.4f} | val {scores['val']:.4f} | lr {lr:.6f} | {time.time()-start:.1f}s", flush=True)
                if scores['val'] < best_val:
                    best_val = scores['val']
                    bad_evals = 0
                    best_is_current = True
                    save(best_path)
                    print('  Новый лучший checkpoint: checkpoints/pretrain_best.pt')
                else:
                    bad_evals += 1
                    best_is_current = False
                save(last_path)
                if step >= opts['warmup_steps'] and bad_evals >= opts['patience']:
                    print('Остановка: проверочная ошибка долго не улучшается.')
                    break
    except KeyboardInterrupt:
        print('\nПрерывание пользователем. Сохраняю последний checkpoint...')
    finally:
        save(last_path)
        print('Сохранено:', last_path)
        print('Лучшее на проверке:', best_path if best_path.exists() else 'пока нет')


if __name__ == '__main__':
    main()
