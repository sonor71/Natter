"""Оценка без выдуманной метрики 'философичности': сохраним ответы для ручного сравнения."""
from __future__ import annotations
import json
from datetime import datetime
import torch

from natter.settings import ROOT
from chat import load_brain


def main():
    model, tok, device = load_brain()
    cases = (ROOT / 'evaluation_prompts.txt').read_text(encoding='utf-8').splitlines()
    answers = []
    for n, question in enumerate((x.strip() for x in cases if x.strip()), 1):
        prefix = tok.encode(f'ПОЛЬЗОВАТЕЛЬ: {question}\nНАТТЕР: ')
        prefix = [tok.bos_id] + prefix
        prefix = prefix[-(model.context_size - 120):]
        x = torch.tensor([prefix], device=device)
        out = model.generate(x, max_new_tokens=120, temperature=0)
        result = tok.decode(out[0, len(prefix):].tolist()).split('\nПОЛЬЗОВАТЕЛЬ:')[0]
        print(f'{n}. {question}\n   {result}\n')
        answers.append({'question': question, 'answer': result, 'scores': {
            'соответствие_вопросу_1_5': None, 'логика_1_5': None,
            'фактическая_точность_1_5': None, 'глубина_1_5': None}})
    target = ROOT / 'logs' / f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Сохранено для ручной оценки:', target)


if __name__ == '__main__':
    main()
