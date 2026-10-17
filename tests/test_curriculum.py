"""Curriculum quality, split separation, SFT alignment and probe smoke tests."""
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import torch

from natter.curriculum import build, SCENARIOS, write_curriculum
from natter.multiturn import encode_conversations, load_conversations
from probe_curriculum import select_cases, task_check, target_loss


class ToyTokenizer:
    bos_id, eos_id, pad_id = 2, 3, 0

    def encode(self, text):
        return [ord(x) + 8 for x in text]


class CurriculumTests(unittest.TestCase):
    def test_splits_are_disjoint_and_include_editorial_philosophy(self):
        data = build(seed=1337)
        self.assertGreater(len(data["train"]), 350)
        self.assertGreater(len(data["val"]), 120)
        train_scenarios = {r["scenario"] for r in data["train"]}
        val_scenarios = {r["scenario"] for r in data["val"]}
        self.assertFalse(train_scenarios & val_scenarios)
        self.assertIn("philosophy", {r["skill"] for r in data["train"]})
        self.assertIn("philosophy", {r["skill"] for r in data["val"]})
        self.assertEqual(data, build(seed=1337))

    def test_writes_safe_versioned_source_and_loads_multiturn(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = write_curriculum(root)
            self.assertEqual(report["scenarios_train"], 24)
            self.assertGreater(report["train_examples"], 350)
            path = root / "data" / "dialogues" / "curriculum" / "train.jsonl"
            rows = load_conversations(path, min_examples=200)
            self.assertEqual(len(rows), report["train_examples"])
            self.assertTrue(all(len(turns) >= 3 for turns in rows))
            with self.assertRaises(FileExistsError):
                write_curriculum(root)

    def test_all_examples_encodable_and_have_user_mask(self):
        tok = ToyTokenizer()
        data = build(seed=1337)
        encoded, skipped = encode_conversations(
            [row["turns"] for row in data["train"][:25]],
            tok, limit=3000
        )
        self.assertEqual(skipped, 0)
        self.assertEqual(len(encoded), 25)
        for xs, ys in encoded:
            self.assertEqual(len(xs), len(ys))
            self.assertEqual(ys[-1], tok.eos_id)
            self.assertIn(-100, ys)
            self.assertTrue(any(y != -100 for y in ys))

    def test_validation_probe_balances_skills_without_scenario_reuse(self):
        cases = select_cases(build()["val"], per_skill=2, seed=42)
        by_skill = Counter(x["skill"] for x in cases)
        self.assertTrue(all(n == 2 for n in by_skill.values()))
        self.assertEqual(len(by_skill), 6)
        self.assertEqual(len({(x["skill"], x["scenario"]) for x in cases}), len(cases))
        philosophic = next(x for x in cases if x["skill"] == "philosophy")
        self.assertIsNone(task_check("Может быть, нужно сравнить аргументы.", philosophic))

    def test_formal_probe_matches_reference_answer(self):
        one = next(x for x in build()["val"] if x["skill"] == "chain")
        self.assertTrue(task_check(one["turns"][-1]["assistant"], one))
        self.assertFalse(task_check("Совсем другой ответ.", one))

    def test_reference_loss_offsets_answer_tokens(self):
        class PerfectToyModel:
            context_size = 16

            def __call__(self, x):
                logits = torch.full((*x.shape, 8), -12.0)
                logits.scatter_(-1, ((x+1) % 8).unsqueeze(-1), 12.0)
                return logits, None

        self.assertLess(
            target_loss(PerfectToyModel(), [1, 2], [3, 4, 5], "cpu"), 0.001
        )
        with self.assertRaises(ValueError):
            target_loss(PerfectToyModel(), [1]*16, [2, 3], "cpu")


if __name__ == "__main__":
    unittest.main()
