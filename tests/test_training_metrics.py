"""Metrics and read-only corpus checks."""
import tempfile
import unittest
from pathlib import Path

from natter.training_metrics import effective_tokens_per_step, training_progress
from audit_corpus import document_category, classify_line, audit_documents


class MetricsTests(unittest.TestCase):
    def test_training_positions_count_repeats(self):
        tokens = effective_tokens_per_step(4, 8, 1024)
        self.assertEqual(tokens, 32768)
        info = training_progress(3200, 1700, tokens, 12_000_000, 3.5565)
        self.assertEqual(info["run_tokens_seen"], 1500 * 32768)
        self.assertAlmostEqual(info["sampled_corpus_passes_this_run"], 4.096)
        self.assertGreater(info["val_perplexity"], 34)
        self.assertLess(info["val_perplexity"], 36)

    def test_invalid_values(self):
        with self.assertRaises(ValueError):
            effective_tokens_per_step(0, 1, 1024)
        with self.assertRaises(ValueError):
            training_progress(2, 5, 1024, 10_000, 3.5)

    def test_category_and_markup(self):
        self.assertEqual(document_category("data/raw/wikipedia_dump/article.txt"), "wikipedia_dump")
        self.assertEqual(document_category("data/raw/preamble.txt"), "(raw root)")
        self.assertIn("image_markup", classify_line("thumb|right|200px"))
        self.assertIn("unexpanded_template", classify_line("{{Template}}"))
        self.assertFalse(classify_line("Ницше рассуждает о морали."))

    def test_audit_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            filename = "data/raw/philosophy/sample.txt"
            path = root / filename
            path.parent.mkdir(parents=True)
            path.write_text("Ницше писал о морали.\n{{мусор}}\n", encoding="utf-8")
            summary = audit_documents([{
                "path": filename, "split": "train", "chars": 35,
            }], root=root)
            self.assertEqual(summary["categories"]["philosophy"]["files"], 1)
            self.assertEqual(summary["documents_with_markers"]["unexpanded_template"], 1)
            self.assertEqual(summary["missing_count"], 0)


if __name__ == "__main__":
    unittest.main()
