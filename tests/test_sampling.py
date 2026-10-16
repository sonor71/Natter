"""Regression tests for repetitive-token decoding (no downloaded models)."""
import types
import unittest

import torch

from natter.model import NatterTransformer
from natter.sampling import block_repeated_ngrams, nucleus_filter, penalize_repeated_tokens


class SamplingTests(unittest.TestCase):
    def test_sign_aware_repetition_penalty(self):
        logits = torch.tensor([[6.0, -6.0, 1.0]])
        generated = torch.tensor([[0, 1, 0]], dtype=torch.long)
        out = penalize_repeated_tokens(logits, generated, 2.0)
        self.assertEqual(out[0, 0].item(), 3.0)
        self.assertEqual(out[0, 1].item(), -12.0)
        self.assertEqual(out[0, 2].item(), 1.0)

    def test_ngram_blocks_only_repeated_continuation(self):
        logits = torch.zeros((1, 10))
        # After [3,3], choosing 3 would repeat bigram (3,3).
        block_repeated_ngrams(logits, torch.tensor([[3, 3]]), 2)
        self.assertEqual(logits[0, 3].item(), float("-inf"))
        self.assertTrue(torch.isfinite(logits[0, 4]))

    def test_nucleus_filter_keeps_one_token(self):
        logits = torch.tensor([[10.0, 0.0, -1.0]])
        filtered = nucleus_filter(logits, 0.5)
        self.assertTrue(torch.isfinite(filtered[0, 0]))
        self.assertTrue(torch.isinf(filtered[0, 1]))
        self.assertTrue(torch.isinf(filtered[0, 2]))

    def test_model_repetition_is_interrupted_with_control(self):
        model = NatterTransformer(vocab_size=12, context_size=16, dim=32,
                                   n_layers=1, n_heads=4, dropout=0)
        def fixed_logits(self, tokens, targets=None):
            logits = torch.zeros((tokens.shape[0], tokens.shape[1], 12))
            logits[..., 7] = 10
            logits[..., 8] = 9
            return logits, None
        model.forward = types.MethodType(fixed_logits, model)
        prompt = torch.tensor([[2, 4]], dtype=torch.long)
        baseline = model.generate(prompt, max_new_tokens=3, temperature=0)
        fixed = model.generate(prompt, max_new_tokens=3, temperature=0,
                               no_repeat_ngram_size=2)
        self.assertEqual(baseline[0, 2:].tolist(), [7, 7, 7])
        self.assertEqual(fixed[0, 2:].tolist(), [7, 7, 8])

    def test_invalid_options_rejected(self):
        model = NatterTransformer(vocab_size=12, context_size=16, dim=32,
                                   n_layers=1, n_heads=4, dropout=0)
        prompt = torch.tensor([[2]], dtype=torch.long)
        for options in (
            {"repetition_penalty": 0.5},
            {"no_repeat_ngram_size": 1},
            {"top_p": 0},
            {"top_p": 2.0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                model.generate(prompt, max_new_tokens=1, **options)


if __name__ == "__main__":
    unittest.main()
