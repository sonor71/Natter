"""Локальный BM25-поиск по собственным текстам, без чужих нейросетей."""
from __future__ import annotations
import json
import math
import re
from collections import Counter
from pathlib import Path

from .settings import ROOT


def terms(text):
    return re.findall(r'[^\W_]+', text.casefold(), flags=re.UNICODE)


def split_into_passages(text, limit=850, overlap=120):
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    pieces = []
    for paragraph in paragraphs:
        if len(paragraph) <= limit:
            pieces.append(paragraph)
        else:
            start = 0
            while start < len(paragraph):
                chunk = paragraph[start:start + limit]
                pieces.append(chunk)
                if start + limit >= len(paragraph):
                    break
                start += limit - overlap
    return pieces


def build_library():
    books = ROOT / 'library'
    rows = []
    for path in sorted(books.rglob('*.txt')):
        text = path.read_text(encoding='utf-8-sig')
        for i, passage in enumerate(split_into_passages(text), 1):
            if len(passage) >= 50:
                rows.append({'source': str(path.relative_to(ROOT)), 'passage': i, 'text': passage})
    destination = ROOT / 'artifacts' / 'library.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
    print(f'Проиндексировано {len(rows)} фрагментов из {len(set(r["source"] for r in rows))} файлов.')
    return rows


class Library:
    def __init__(self):
        path = ROOT / 'artifacts' / 'library.json'
        self.rows = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
        self.counts = [Counter(terms(r['text'])) for r in self.rows]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.avg_length = sum(self.lengths) / max(1, len(self.lengths))
        self.df = Counter()
        for c in self.counts:
            self.df.update(c.keys())

    def search(self, query, k=2):
        requested = set(terms(query))
        n = len(self.rows)
        scores = []
        for i, cnt in enumerate(self.counts):
            score = 0.0
            length = self.lengths[i]
            for term in requested:
                if term not in cnt:
                    continue
                df = self.df[term]
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                freq = cnt[term]
                score += idf * (freq * 2.2) / (freq + 1.2 * (0.25 + 0.75 * length / self.avg_length))
            if score > 0:
                scores.append((score, self.rows[i]))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [dict(row, score=round(score, 3)) for score, row in scores[:k]]
