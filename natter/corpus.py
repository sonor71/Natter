"""Disk-streamed corpus preparation for NATTER CORE.

The trainer's binary format (little-endian uint32 tokens) and manifest keys
remain compatible with NATTER CORE 1.0.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .settings import ROOT, sha256_file
from .tokenizer import NatterTokenizer, train_tokenizer

MIN_DOCUMENT_CHARS = 120
CHUNK_CHARS = 16_384
MIN_PARAGRAPH_CHARS = 120


def _clean_line(line: str) -> str:
    return line.replace('\r', '').replace('\x00', '').rstrip() + '\n'


def scan_document(path: Path) -> dict:
    digest = hashlib.sha256()
    chars = 0
    bytes_utf8 = 0
    paragraphs_seen = set()
    paragraphs = 0
    repeated_paragraphs = 0
    paragraph_digest = hashlib.sha256()
    paragraph_chars = 0

    def finish_paragraph():
        nonlocal paragraphs, repeated_paragraphs, paragraph_digest, paragraph_chars
        if paragraph_chars >= MIN_PARAGRAPH_CHARS:
            fingerprint = paragraph_digest.digest()
            paragraphs += 1
            if fingerprint in paragraphs_seen:
                repeated_paragraphs += 1
            else:
                paragraphs_seen.add(fingerprint)
        paragraph_digest = hashlib.sha256()
        paragraph_chars = 0

    try:
        with path.open('r', encoding='utf-8-sig') as fh:
            for raw in fh:
                line = _clean_line(raw)
                chars += len(line)
                bytes_utf8 += len(line.encode('utf-8'))
                normalized = ' '.join(line.split())
                if normalized:
                    digest.update(normalized.encode('utf-8'))
                    digest.update(b' ')
                    paragraph_digest.update(normalized.encode('utf-8'))
                    paragraph_digest.update(b' ')
                    paragraph_chars += len(normalized) + 1
                else:
                    finish_paragraph()
        finish_paragraph()
    except UnicodeDecodeError as e:
        raise RuntimeError(f'Не UTF-8 файл: {path}. Пересохрани его как UTF-8.') from e
    return {
        'path': str(path.relative_to(ROOT)).replace('\\', '/'),
        'sha256': digest.hexdigest(),
        'chars': chars,
        'utf8_bytes': bytes_utf8,
        'paragraphs': paragraphs,
        'repeated_paragraphs': repeated_paragraphs,
    }


def discover_documents(raw_dir: Path):
    paths = sorted(raw_dir.rglob('*.txt'))
    if not paths:
        raise RuntimeError(f'Добавь UTF-8 .txt файлы в {raw_dir}')
    docs, fingerprints, duplicates, too_short = [], {}, [], []
    for path in paths:
        doc = scan_document(path)
        if doc['chars'] < MIN_DOCUMENT_CHARS:
            too_short.append(doc['path'])
            continue
        if doc['sha256'] in fingerprints:
            duplicates.append({'path': doc['path'], 'duplicate_of': fingerprints[doc['sha256']]})
            continue
        fingerprints[doc['sha256']] = doc['path']
        docs.append(doc)
    if len(docs) < 3:
        raise RuntimeError('Нужно минимум 3 уникальных документа размером от 120 символов.')
    return docs, duplicates, too_short


def split_documents(docs, fraction: float, seed: int):
    if not 0 < fraction < 1:
        raise ValueError('validation_fraction должно быть от 0 до 1.')
    # Stable per-path hash: newly added documents do not randomly reshuffle old ones.
    docs = sorted(docs, key=lambda d: d['path'])
    val, train = [], []
    for doc in docs:
        key = f'{seed}/{doc["path"]}'.encode('utf-8')
        score = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') / 2**64
        (val if score < fraction else train).append(doc)
    if not val:
        val.append(train.pop())
    if not train:
        train.append(val.pop())
    return train, val


def read_clean_lines(doc):
    with (ROOT / doc['path']).open('r', encoding='utf-8-sig') as fh:
        for raw in fh:
            yield _clean_line(raw)


def write_tokenizer_samples(train_docs, sample_file: Path, max_chars: int) -> int:
    """Bound tokenizer samples by size, and sample from every training document."""
    per_doc_budget = max(256, max_chars // len(train_docs))
    total = 0
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    with sample_file.open('w', encoding='utf-8') as out:
        for doc in train_docs:
            used = 0
            for line in read_clean_lines(doc):
                if used >= per_doc_budget or total >= max_chars:
                    break
                line = line.strip()
                if not line:
                    continue
                allowed = min(len(line), per_doc_budget - used, max_chars - total)
                if allowed <= 0:
                    break
                snippet = line[:allowed]
                if snippet:
                    out.write(snippet + '\n')
                    used += len(snippet)
                    total += len(snippet)
    return total


def encode_document(doc, tok: NatterTokenizer, fh) -> int:
    count = 0

    def emit(ids):
        nonlocal count
        np.asarray(ids, dtype='<u4').tofile(fh)
        count += len(ids)

    emit([tok.bos_id])
    buffer = []
    buf_len = 0
    for line in read_clean_lines(doc):
        # Split very long lines so that RAM consumption is bounded.
        for i in range(0, len(line), CHUNK_CHARS):
            segment = line[i:i + CHUNK_CHARS]
            if buf_len + len(segment) > CHUNK_CHARS and buffer:
                emit(tok.encode(''.join(buffer)))
                buffer, buf_len = [], 0
            buffer.append(segment)
            buf_len += len(segment)
    if buffer:
        emit(tok.encode(''.join(buffer)))
    emit([tok.eos_id])
    return count


def build_corpus(cfg, retrain_tokenizer=False, dry_run=False, max_tokenizer_chars=20_000_000):
    processed = ROOT / 'data' / 'processed'
    processed.mkdir(parents=True, exist_ok=True)
    tokenizer_path = ROOT / 'artifacts' / 'tokenizer.model'
    documents, duplicates, too_short = discover_documents(ROOT / 'data' / 'raw')
    train_docs, val_docs = split_documents(documents, cfg['validation_fraction'], cfg['training']['seed'])
    repeat_warnings = [d for d in documents if d['paragraphs'] >= 5 and d['repeated_paragraphs'] / d['paragraphs'] > 0.2]
    report = {
        'unique_documents': len(documents),
        'train_documents': len(train_docs), 'validation_documents': len(val_docs),
        'unique_chars': sum(d['chars'] for d in documents),
        'unique_utf8_bytes': sum(d['utf8_bytes'] for d in documents),
        'exact_duplicates': duplicates,
        'too_short': too_short,
        'high_internal_repetition': repeat_warnings,
        'documents': [{**d, 'split': 'train' if d in train_docs else 'val'} for d in documents],
    }
    (processed / 'corpus_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Уникальных файлов: {len(documents)} | train: {len(train_docs)} | val: {len(val_docs)}")
    print(f"Символов: {report['unique_chars']:,}; точных копий: {len(duplicates)}; коротких: {len(too_short)}")
    if repeat_warnings:
        print(f'ВНИМАНИЕ: в {len(repeat_warnings)} файлах много повторённых абзацев. См. data/processed/corpus_report.json')
    if dry_run:
        print('Предварительная проверка завершена: токенизатор и .bin не изменены.')
        return report

    if retrain_tokenizer and any((ROOT / 'checkpoints').glob('*.pt')):
        raise RuntimeError('Нельзя менять токенизатор, пока в checkpoints/ есть .pt: несовместимые ID токенов. Сначала сохрани архив старых весов.')
    if not tokenizer_path.exists() or retrain_tokenizer:
        sample_path = processed / 'tokenizer_sample.txt'
        sample_chars = write_tokenizer_samples(train_docs, sample_path, max_tokenizer_chars)
        print(f'Обучаю BPE на выборке из TRAIN: {sample_chars:,} символов (VAL не используется).')
        train_tokenizer(sample_path, tokenizer_path.with_suffix(''), cfg['tokenizer_vocab_size'])
        sample_path.unlink(missing_ok=True)

    tok = NatterTokenizer(tokenizer_path)
    counts = {}
    for split, group in [('train', train_docs), ('val', val_docs)]:
        target = processed / (split + '.bin')
        count = 0
        with target.open('wb') as fh:
            for i, doc in enumerate(group, 1):
                count += encode_document(doc, tok, fh)
                if i % 100 == 0:
                    print(f'  {split}: {i}/{len(group)} файлов, {count:,} токенов', flush=True)
        counts[split] = count

    identity = json.dumps(
        sorted((d['path'], d['sha256'], 'train' if d in train_docs else 'val') for d in documents),
        ensure_ascii=False, separators=(',', ':')
    )
    manifest = {
        'tokenizer_sha256': sha256_file(tokenizer_path),
        'corpus_sha256': hashlib.sha256(identity.encode('utf-8')).hexdigest(),
        'train_documents': [d['path'] for d in train_docs],
        'val_documents': [d['path'] for d in val_docs],
        'token_counts': counts,
        'skipped_short_or_duplicate': len(duplicates) + len(too_short),
    }
    (processed / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Токены: train {counts['train']:,}, val {counts['val']:,} | BPE {tok.vocab_size:,}")
    print('Подготовлено: data/processed/train.bin, val.bin, manifest.json, corpus_report.json')
    if min(counts.values()) <= cfg['model']['context_size'] + 1:
        print('ВНИМАНИЕ: для окна модели мало проверочных или обучающих токенов. Добавь документы.')
    return manifest


def load_manifest():
    path = ROOT / 'data' / 'processed' / 'manifest.json'
    if not path.exists():
        raise RuntimeError('Сначала запусти: python prepare.py')
    return json.loads(path.read_text(encoding='utf-8'))


class StreamData:
    def __init__(self, split):
        path = ROOT / 'data' / 'processed' / (split + '.bin')
        if not path.exists():
            raise RuntimeError('Сначала запусти: python prepare.py')
        self.tokens = np.memmap(path, dtype='<u4', mode='r')

    def batch(self, batch_size, context_size, device, generator=None):
        import torch
        if len(self.tokens) <= context_size + 1:
            raise RuntimeError(f'Недостаточно токенов ({len(self.tokens)}) для context_size={context_size}.')
        # Optional independent RNG lets validation use the same windows each time.
        starts = torch.randint(0, len(self.tokens) - context_size, (batch_size,), generator=generator).tolist()
        chunks = np.stack([self.tokens[i:i + context_size + 1] for i in starts]).astype('int64')
        tokens = torch.from_numpy(chunks)
        return tokens[:, :-1].to(device), tokens[:, 1:].to(device)
