"""Tests for comparable validation batches and reference continuation loss."""
import unittest

import numpy as np
import torch

from natter.corpus import StreamData
from probe_language import evaluate_continuation


class CorpusEvalTests(unittest.TestCase):
    def test_batches_repeat_with_same_generator_seed(self):
        stream = StreamData.__new__(StreamData)
        stream.tokens = np.arange(5000, dtype="<u4")
        seed = 1337
        a_x, a_y = stream.batch(8, 32, "cpu", generator=torch.Generator().manual_seed(seed))
        b_x, b_y = stream.batch(8, 32, "cpu", generator=torch.Generator().manual_seed(seed))
        self.assertTrue(torch.equal(a_x, b_x))
        self.assertTrue(torch.equal(a_y, b_y))
        self.assertTrue(torch.equal(a_y, a_x + 1))

    def test_random_training_batches_without_generator_still_work(self):
        stream = StreamData.__new__(StreamData)
        stream.tokens = np.arange(2000, dtype="<u4")
        x, y = stream.batch(4, 16, "cpu")
        self.assertEqual(tuple(x.shape), (4, 16))
        self.assertEqual(tuple(y.shape), (4, 16))

    def test_reference_loss_uses_only_continuation(self):
        class PerfectToyModel:
            context_size = 16

            def __call__(self, x):
                logits = torch.full((*x.shape, 8), -10.0)
                expected = ((x + 1) % 8).unsqueeze(-1)
                logits.scatter_(-1, expected, 10.0)
                return logits, None

        loss = evaluate_continuation(
            PerfectToyModel(), prompt_ids=[1, 2], continuation_ids=[3, 4, 5], device="cpu"
        )
        self.assertLess(loss, 0.0001)

    def test_reference_loss_rejects_context_overflow(self):
        class Tiny:
            context_size = 3

        with self.assertRaises(ValueError):
            evaluate_continuation(Tiny(), [1, 2, 3], [4, 5], "cpu")


if __name__ == "__main__":
    unittest.main()
