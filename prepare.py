"""Prepare a training/validation corpus without loading whole documents in RAM."""
import argparse
from natter.corpus import build_corpus
from natter.settings import config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NATTER: streaming corpus + BPE')
    parser.add_argument('--dry-run', action='store_true', help='Анализ файлов без создания токенов и BPE')
    parser.add_argument('--retrain-tokenizer', action='store_true', help='Создать новый BPE (не совместим со старыми весами)')
    parser.add_argument('--max-tokenizer-chars', type=int, default=20_000_000, help='Лимит символов в выборке для BPE')
    args = parser.parse_args()
    if args.max_tokenizer_chars < 5000:
        parser.error('--max-tokenizer-chars должно быть не меньше 5000')
    build_corpus(config(), retrain_tokenizer=args.retrain_tokenizer,
                 dry_run=args.dry_run, max_tokenizer_chars=args.max_tokenizer_chars)
