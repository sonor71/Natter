"""Tokenizer smoke tests using only a temporary locally trained SentencePiece BPE."""
import tempfile
import unittest
from pathlib import Path

from natter.tokenizer import NatterTokenizer, train_tokenizer


class TokenizerTests(unittest.TestCase):
    def test_russian_roundtrip_and_byte_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            train_file = root / "sample.txt"
            text = "Привет, мир! Наттер читает Ницше и Батая."
            train_file.write_text((text + "\n") * 120, encoding="utf-8")
            model_file = train_tokenizer(train_file, root / "test", 512)
            tokenizer = NatterTokenizer(model_file)
            ids = tokenizer.encode(text)
            self.assertEqual(tokenizer.decode(ids), text)
            self.assertNotIn(tokenizer.sp.unk_id(), ids)
            self.assertGreater(tokenizer.vocab_size, 260)


if __name__ == "__main__":
    unittest.main()
