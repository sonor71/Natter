"""CPU smoke tests for checkpoint-compatible generation and sampling."""
import unittest

import torch

from natter.model import NatterTransformer


class GenerationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model = NatterTransformer(
            vocab_size=32, context_size=32, dim=32, n_layers=2,
            n_heads=4, dropout=0.0,
        )
        self.model.eval()

    def test_forward_shapes_and_finite_loss(self):
        x = torch.tensor([[3, 4, 5, 6]], dtype=torch.long)
        logits, loss = self.model(x, x)
        self.assertEqual(tuple(logits.shape), (1, 4, 32))
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertTrue(torch.isfinite(loss).item())

    def test_forbidden_special_tokens_not_generated(self):
        x = torch.tensor([[3, 4]], dtype=torch.long)
        torch.manual_seed(3)
        result = self.model.generate(
            x, max_new_tokens=8, temperature=0.8, top_k=10,
            forbidden_token_ids=(0, 1, 2), eos_id=None,
        )
        self.assertEqual(tuple(result.shape), (1, 10))
        self.assertTrue(all(i not in (0, 1, 2) for i in result[0, 2:].tolist()))

    def test_greedy_is_repeatable(self):
        x = torch.tensor([[3, 4, 5]], dtype=torch.long)
        a = self.model.generate(x, max_new_tokens=5, temperature=0)
        b = self.model.generate(x, max_new_tokens=5, temperature=0)
        self.assertTrue(torch.equal(a, b))

    def test_invalid_prefix_and_full_token_ban(self):
        with self.assertRaises(ValueError):
            self.model.generate(torch.empty(1, 0, dtype=torch.long))
        with self.assertRaises(ValueError):
            self.model.generate(
                torch.tensor([[3]], dtype=torch.long),
                forbidden_token_ids=range(32),
            )


if __name__ == "__main__":
    unittest.main()
