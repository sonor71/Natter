"""Multi-turn SFT regression tests: context and supervision align by token ID."""
import json
import tempfile
import unittest
from pathlib import Path

from natter.multiturn import (
    ASSISTANT, USER, build_context, encode_conversation,
    encode_conversations, load_conversations,
)


class ToyTokenizer:
    bos_id = 2
    eos_id = 3
    pad_id = 0

    def encode(self, text):
        # An intentionally lossless simple toy encoder with no BPE dependencies.
        return [ord(c) + 8 for c in text]


class MultiTurnTests(unittest.TestCase):
    def setUp(self):
        self.tok = ToyTokenizer()
        self.turns = [
            {"user": "Что такое свобода?", "assistant": "Возможность выбора."},
            {"user": "А что из этого следует?", "assistant": "Выбор предполагает альтернативы."},
        ]

    def test_encoded_labels_mask_every_user_and_supervise_answers(self):
        x, y = encode_conversation(self.turns, self.tok, limit=1024)
        full = [self.tok.bos_id] + [
            x for turn in self.turns
            for x in (
                self.tok.encode(f"{USER}{turn['user']}{ASSISTANT}") +
                self.tok.encode(turn["assistant"]) + [self.tok.eos_id]
            )
        ]
        self.assertEqual(x, full[:-1])
        self.assertEqual(len(x), len(y))
        # Last target for each assistant segment is EOS.
        self.assertEqual(y[-1], self.tok.eos_id)
        self.assertEqual(sum(t == self.tok.eos_id for t in y), 2)
        first_answer_start = 1 + len(self.tok.encode(f"{USER}{self.turns[0]['user']}{ASSISTANT}"))
        self.assertTrue(all(z == -100 for z in y[:first_answer_start - 1]))
        self.assertEqual(y[first_answer_start-1],
                         self.tok.encode(self.turns[0]['assistant'])[0])
        second_question_start = first_answer_start + len(self.tok.encode(self.turns[0]["assistant"])) + 1
        second_question_end = second_question_start + len(self.tok.encode(
            f"{USER}{self.turns[1]['user']}{ASSISTANT}"
        ))
        self.assertTrue(all(z == -100 for z in y[second_question_start - 1:second_question_end - 1]))

    def test_inference_prefix_matches_training_prefix(self):
        expected = (
            [self.tok.bos_id] +
            self.tok.encode(f"{USER}{self.turns[0]['user']}{ASSISTANT}") +
            self.tok.encode(self.turns[0]['assistant']) + [self.tok.eos_id] +
            self.tok.encode(f"{USER}{self.turns[1]['user']}{ASSISTANT}")
        )
        actual = build_context(
            self.tok, [(self.turns[0]['user'], self.turns[0]['assistant'])],
            self.turns[1]['user'], context_size=1024, output_budget=60
        )
        self.assertEqual(expected, actual)

    def test_context_drops_older_turns_to_fit(self):
        history = [("старое сообщение", "старый ответ"), ("новый вопрос", "новый ответ")]
        prefix = build_context(self.tok, history, "продолжай", context_size=145, output_budget=20)
        self.assertLessEqual(len(prefix), 125)
        self.assertEqual(prefix[0], self.tok.bos_id)
        self.assertEqual(prefix[-1], self.tok.encode(ASSISTANT)[-1])

    def test_long_example_skipped(self):
        encoded, skipped = encode_conversations([self.turns], self.tok, limit=32)
        self.assertEqual(encoded, [])
        self.assertEqual(skipped, 1)

    def test_load_rejects_incorrect_records(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.jsonl"
            path.write_text(json.dumps({"turns": [{"user": "A", "assistant": "B"}]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_conversations(path, min_examples=1)

    def test_load_two_valid_records(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.jsonl"
            path.write_text(
                "\n".join(json.dumps({"turns": self.turns}, ensure_ascii=False) for _ in range(2)),
                encoding="utf-8"
            )
            self.assertEqual(len(load_conversations(path)), 2)


if __name__ == "__main__":
    unittest.main()
