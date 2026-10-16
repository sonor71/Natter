import json
import tempfile
import unittest
from pathlib import Path

import torch

from natter.model import NatterTransformer
from natter.library import Library, terms, split_into_passages
from natter.memory import Memory
from natter.dialogues import encode_dialogues


class ModelTests(unittest.TestCase):
    def test_forward_and_backward(self):
        model = NatterTransformer(vocab_size=300, context_size=32, dim=48, n_layers=2, n_heads=4)
        x = torch.randint(0, 300, (2, 12))
        y = torch.randint(0, 300, (2, 12))
        logits, loss = model(x, y)
        self.assertEqual(logits.shape, (2, 12, 300))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.embed.weight.grad)

    def test_causality(self):
        model = NatterTransformer(vocab_size=300, context_size=32, dim=48, n_layers=1, n_heads=4)
        model.eval()
        a = torch.tensor([[2, 13, 9, 10, 3]])
        b = torch.tensor([[2, 13, 9, 100, 3]])
        with torch.no_grad():
            aa = model(a)[0]
            bb = model(b)[0]
        self.assertTrue(torch.allclose(aa[:, :3], bb[:, :3], atol=1e-5))

    def test_generate_greedy(self):
        model = NatterTransformer(vocab_size=300, context_size=32, dim=48, n_layers=1, n_heads=4)
        x = torch.tensor([[2, 10, 11]])
        out = model.generate(x, max_new_tokens=2, temperature=0)
        self.assertEqual(out.shape, (1, 5))

    def test_tokenizer_mask(self):
        class DummyTok:
            bos_id, eos_id = 2, 3
            def encode(self, x):
                return [4 + ord(c) % 50 for c in x]
        sample, skipped = encode_dialogues([('Привет', 'Ответ')], DummyTok(), 120)
        self.assertEqual(skipped, 0)
        self.assertTrue(-100 in sample[0][1])
        self.assertTrue(any(v >= 0 for v in sample[0][1]))

    def test_library_terms(self):
        self.assertIn('ницше', terms('Работы Ницше, а также Батая.'))
        self.assertEqual(split_into_passages('Первый\n\nВторой'), ['Первый', 'Второй'])


if __name__ == '__main__':
    unittest.main()
