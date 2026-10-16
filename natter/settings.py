from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def device_name():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def model_config(cfg, vocab_size):
    return {'vocab_size': vocab_size, **cfg['model']}
