"""SentencePiece BPE, with a local training corpus; no downloaded pretrained weights."""
from __future__ import annotations
from pathlib import Path
import sentencepiece as spm


class NatterTokenizer:
    def __init__(self, model_file):
        self.model_file = Path(model_file)
        self.sp = spm.SentencePieceProcessor(model_file=str(self.model_file))
        self.vocab_size = self.sp.vocab_size()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id()

    def encode(self, text: str):
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        return self.sp.decode([int(x) for x in ids if int(x) not in (self.bos_id, self.eos_id, self.pad_id)])


def train_tokenizer(training_file: Path, model_prefix: Path, vocab_size: int):
    """Train BPE from the bounded file prepared from training documents only."""
    model_prefix = str(model_prefix)
    Path(model_prefix).parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(training_file), model_prefix=model_prefix,
        vocab_size=vocab_size, model_type='bpe',
        character_coverage=1.0, byte_fallback=True,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        hard_vocab_limit=False,
        max_sentence_length=32768, minloglevel=2,
    )
    return Path(model_prefix + '.model')
