"""Tests for the fast CC0 dialogue import and single-turn SFT prompt."""
import json
import random
import tempfile
import unittest
from pathlib import Path

from import_dialogues import answer_key, group_val, select_pairs, is_usable
from natter.dialogues import load_jsonl_pairs, sample_batch, encode_dialogues


class DialogueImportTests(unittest.TestCase):
    def test_split_is_disjoint_by_exact_answer_and_repetition_is_capped(self):
        rows = [
            {"q": f"Что ты думаешь о теме {a} вариант {b}?",
             "a": f"Это интересная тема номер {a}; изучить её стоит подробнее."}
            for a in range(150) for b in range(6)
        ]
        train, val, stats = select_pairs(rows, max_per_answer=2, seed=42)
        self.assertLessEqual(len(train) + len(val), 300)
        self.assertEqual(stats["distinct_answer_groups"], 150)
        self.assertFalse({answer_key(a) for _, a in train} &
                         {answer_key(a) for _, a in val})
        self.assertGreater(len(train), 10)
        self.assertGreater(len(val), 5)
        self.assertEqual((train, val), select_pairs(rows, max_per_answer=2, seed=42)[:2])

    def test_exclude_other_assistant_identity_and_bad_markup(self):
        self.assertFalse(is_usable("Где ты?", "Я живу в облаке."))
        self.assertFalse(is_usable("Как дела?", "{{Some template}}"))
        self.assertTrue(is_usable("Что такое смысл?", "Смысл зависит от выбранных целей."))

    def test_load_external_pairs_without_touching_approved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "train.jsonl"
            path.write_text(
                json.dumps({"user": "Что такое свобода?", "assistant": "Возможность выбора."},
                           ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_jsonl_pairs(path, min_examples=1), [
                ("Что такое свобода?", "Возможность выбора.")
            ])

    def test_mask_prompt_and_deterministic_batches(self):
        class ToyTokenizer:
            bos_id, eos_id, pad_id = 2, 3, 0

            def encode(self, text):
                return [4 + (ord(char) % 64) for char in text]

        tok = ToyTokenizer()
        encoded, skipped = encode_dialogues(
            [("Привет", "Здравствуйте"), ("Как дела?", "Нормально")],
            tok, 128,
        )
        self.assertEqual(skipped, 0)
        prefix_len = 1 + len(tok.encode("ПОЛЬЗОВАТЕЛЬ: Привет\nНАТТЕР: "))
        self.assertTrue(all(t == -100 for t in encoded[0][1][:prefix_len-1]))
        self.assertTrue(any(t != -100 for t in encoded[0][1][prefix_len-1:]))
        import torch
        x1, y1 = sample_batch(encoded, 2, 0, "cpu", rng=random.Random(44))
        x2, y2 = sample_batch(encoded, 2, 0, "cpu", rng=random.Random(44))
        self.assertTrue(torch.equal(x1, x2))
        self.assertTrue(torch.equal(y1, y2))


if __name__ == "__main__":
    unittest.main()
