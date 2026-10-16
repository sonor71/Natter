"""Атомарное сохранение локальных checkpoint. Не загружайте непроверенные .pt."""
from pathlib import Path
import torch


def save_checkpoint(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, device='cpu'):
    return torch.load(path, map_location=device, weights_only=True)


def make_model(cfg, tok, device):
    from .model import NatterTransformer
    from .settings import model_config
    kwargs = model_config(cfg, tok.vocab_size)
    return NatterTransformer(**kwargs).to(device), kwargs


def validate_checkpoint(ckpt, expected_arch, tokenizer_sha256):
    if ckpt.get('arch') != expected_arch:
        raise RuntimeError('Архитектура checkpoint не совпадает с config.json: проверь размер, слои, контекст или токенизатор.')
    if ckpt.get('tokenizer_sha256') != tokenizer_sha256:
        raise RuntimeError('Токенизатор изменился! Нельзя продолжать обучение старых весов с изменёнными ID токенов.')
