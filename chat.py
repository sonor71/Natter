from __future__ import annotations
import argparse
import json
from datetime import datetime

import torch

from natter.checkpoints import load_checkpoint, make_model, validate_checkpoint
from natter.corpus import load_manifest
from natter.library import Library
from natter.argument_logic import ArgumentSyntaxError, reason
from natter.multiturn import build_context
from natter.memory import Memory
from natter.settings import ROOT, config, device_name, sha256_file
from natter.tokenizer import NatterTokenizer


def select_checkpoint(stage):
    names = (['dialogue_multiturn_best.pt', 'dialogue_tiny_chat_best.pt', 'dialogue_best.pt', 'dialogue_tiny_ru_best.pt', 'pretrain_best.pt'] if stage == 'auto' else
             ['pretrain_best.pt'] if stage == 'base' else
             ['dialogue_tiny_chat_best.pt'] if stage == 'tiny-chat' else
             ['dialogue_multiturn_best.pt'] if stage == 'multiturn' else
             ['dialogue_best.pt', 'dialogue_tiny_chat_best.pt', 'dialogue_tiny_ru_best.pt'])
    for name in names:
        path = ROOT / 'checkpoints' / name
        if path.exists():
            return path
    raise RuntimeError('Checkpoint не найден. Сначала подготовь корпус и обучи модель.')


def load_brain(stage='auto'):
    cfg = config()
    device = device_name()
    tok = NatterTokenizer(ROOT / 'artifacts' / 'tokenizer.model')
    model, arch = make_model(cfg, tok, device)
    ck_path = select_checkpoint(stage)
    ck = load_checkpoint(ck_path, device)
    validate_checkpoint(ck, arch, sha256_file(tok.model_file))
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    print(f'Модель: {ck_path.name}; шаг: {ck["step"]}; устройство: {device}')
    if ck.get('stage') == 'pretrain':
        print('ВНИМАНИЕ: это базовая модель продолжения текста, не обученная вести диалог.')
    return model, tok, device


def append_log(path, row):
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def make_prompt(question, history, memory, library, tokenizer, context, budget, dialogue_mode=False):
    if dialogue_mode:
        # Match the prefix used by dialogue supervised fine-tuning exactly.
        # The current SFT dataset contains single-turn Q/A, not system prompts,
        # long chat histories or retrieved book passages.
        pure = f'ПОЛЬЗОВАТЕЛЬ: {question}\nНАТТЕР: '
        limit = max(32, context - budget)
        ids = [tokenizer.bos_id] + tokenizer.encode(pure)
        if len(ids) > limit:
            ids = [ids[0]] + ids[-(limit - 1):]
        return ids, []
    passages = library.search(question, k=1)
    memories = memory.search(question, k=2)
    system = 'Ты — НАТТЕР. Отвечай по существу. Не выдумывай источники и цитаты.\n'
    extras = ''
    if memories:
        extras += 'ПАМЯТЬ:\n' + ''.join(f'- {item[2][:180]}\n' for item in memories)
    if passages:
        p = passages[0]
        extras += f'ИСТОЧНИК ({p["source"]}, фрагмент {p["passage"]}):\n{p["text"][:440]}\n'
    end = f'ПОЛЬЗОВАТЕЛЬ: {question}\nНАТТЕР: '
    conversation = ''.join(f'ПОЛЬЗОВАТЕЛЬ: {q}\nНАТТЕР: {a}\n' for q, a in history[-4:])
    whole = system + extras + conversation + end
    max_prompt = max(32, context - budget)
    ids = [tokenizer.bos_id] + tokenizer.encode(whole)
    # При переполнении сохраняем последний вопрос, затем наиболее свежую историю.
    if len(ids) > max_prompt:
        ids = [ids[0]] + ids[-(max_prompt - 1):]  # сохраняем BOS при обрезке контекста
    return ids, passages


def main():
    parser = argparse.ArgumentParser(description='Локальный текстовый чат NATTER CORE')
    parser.add_argument('--stage', choices=['auto', 'base', 'dialogue', 'tiny-chat', 'multiturn'], default='auto')
    parser.add_argument('--tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=40)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--repetition-penalty', type=float, default=1.15)
    parser.add_argument('--no-repeat-ngram', type=int, default=3)
    parser.add_argument('--seed', type=int, default=None,
                        help='Фиксировать случайность для воспроизводимости')
    args = parser.parse_args()
    if args.top_k < 0 or args.tokens < 1:
        parser.error('--top-k должен быть >= 0; --tokens должен быть >= 1')
    if not 0 < args.top_p <= 1 or args.repetition_penalty < 1 or args.no_repeat_ngram not in (0,) and args.no_repeat_ngram < 2:
        parser.error('--top-p: (0,1]; --repetition-penalty: >=1; --no-repeat-ngram: 0 или >=2')
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    model, tok, device = load_brain(args.stage)
    context = model.context_size
    loaded_name = select_checkpoint(args.stage).name
    is_dialogue = loaded_name.startswith('dialogue')
    is_multiturn = loaded_name.startswith('dialogue_multiturn')
    if is_dialogue:
        print('Диалоговая модель: использую формат вопрос-ответ, на котором она обучалась.')
    if is_multiturn:
        print('Многотуровый режим: передаю модели предыдущие 4 реплики, с EOS между ответами.')
    memory, library = Memory(), Library()
    (ROOT / 'logs').mkdir(exist_ok=True)
    logfile = ROOT / 'logs' / f'chat_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'
    history = []
    last_question = None
    print('Команды: /exit, /fix <правильный ответ>, /remember <факт>,')
    print(' /logic <посылки через ;> — строгий многошаговый вывод (без модели)')
    print(' /position <тезис> | <аргументы> | <сомнения>, /mem, /find <тема>')
    print('Разговор записывается в:', logfile)
    while True:
        try:
            question = input('\nТы: ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nВыход')
            break
        if not question:
            continue
        if question in ('/exit', '/quit', 'выход'):
            break
        if question == '/logic' or question.startswith('/logic '):
            try:
                print(reason(question[6:].strip()).render())
            except ArgumentSyntaxError as exc:
                print('Ошибка формальной логики:', exc)
            continue
        if question.startswith('/fix '):
            if last_question is None:
                print('Сначала задай обычный вопрос.')
                continue
            corrected = question[5:].strip()
            if not corrected:
                continue
            dest = ROOT / 'data' / 'dialogues' / 'approved.jsonl'
            dest.parent.mkdir(parents=True, exist_ok=True)
            append_log(dest, {'user': last_question, 'assistant': corrected})
            print('Исправленный ответ сохранён для отдельного обучения диалогам.')
            continue
        if question.startswith('/remember '):
            memory.add('fact', question[10:].strip())
            print('Факт записан в долговременную память (веса не менялись).')
            continue
        if question.startswith('/position '):
            fields = [part.strip() for part in question[10:].split('|')]
            if len(fields) != 3 or not all(fields):
                print('Формат: /position тезис | основания | сомнения')
                continue
            memory.add('position', 'Тезис: ' + fields[0] + '\nОснования: ' + fields[1] + '\nСомнения: ' + fields[2])
            print('Аргументированная позиция сохранена.')
            continue
        if question == '/mem':
            for item in memory.recent():
                print(f'#{item[0]} {item[1]}: {item[2]}')
            continue
        if question.startswith('/find '):
            items = library.search(question[6:], k=3)
            if not items:
                print('В библиотеке ничего подходящего не найдено.')
            for item in items:
                print(f'[{item["source"]} §{item["passage"]}] {item["text"][:650]}')
            continue
        budget = max(1, min(args.tokens, context // 3))
        if is_multiturn:
            ids = build_context(tok, history, question, context, budget, max_turns=4)
            citations = []
        else:
            ids, citations = make_prompt(question, history, memory, library, tok, context, budget,
                                         dialogue_mode=is_dialogue)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        forbidden = [i for i in (tok.pad_id, tok.bos_id, tok.sp.unk_id()) if i >= 0]
        generated = model.generate(x, max_new_tokens=budget, temperature=args.temperature,
                                   top_k=args.top_k, eos_id=tok.eos_id,
                                   forbidden_token_ids=forbidden,
                                   repetition_penalty=args.repetition_penalty,
                                   no_repeat_ngram_size=args.no_repeat_ngram,
                                   top_p=args.top_p)
        answer = tok.decode(generated[0, len(ids):].tolist()).strip()
        for delim in ('\nПОЛЬЗОВАТЕЛЬ:', '\nНАТТЕР:'):
            answer = answer.split(delim)[0]
        print('Наттер:', answer or '[Пустой ответ: модель ещё недостаточно обучена]')
        if citations:
            print('Найденный источник (не гарантия, что модель его верно использовала):',
                  citations[0]['source'], f'§{citations[0]["passage"]}')
        append_log(logfile, {'user': question, 'assistant': answer,
                             'sources': [f'{p["source"]} §{p["passage"]}' for p in citations]})
        last_question = question
        history.append((question, answer))


if __name__ == '__main__':
    main()
