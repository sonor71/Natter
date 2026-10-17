"""Proof-level regression checks: evidence-linked rules, not fabricated reasoning."""
import unittest

from natter.argument_logic import ArgumentSyntaxError, reason


class FormalArgumentTests(unittest.TestCase):
    def test_three_step_reasoning(self):
        result = reason(
            "Если свобода предполагает выбор, то выбор имеет альтернативы; "
            "Если выбор имеет альтернативы, то возможна ответственность; "
            "Если возможна ответственность, то поступок можно оценивать; "
            "свобода предполагает выбор"
        )
        self.assertEqual(len(result.steps), 3)
        self.assertIn("поступок можно оценивать", result.conclusions)
        self.assertFalse(result.contradictions)
        self.assertEqual([s[3] for s in result.steps],
                         ["modus ponens", "modus ponens", "modus ponens"])

    def test_modus_tollens(self):
        result = reason("Если а, то б; не б")
        self.assertIn("не а", result.conclusions)
        self.assertIn("modus tollens", result.render())

    def test_explicit_contradiction_flagged(self):
        result = reason("а -> б; а; не б")
        self.assertTrue(result.contradictions)
        self.assertIn("не а", result.conclusions)

    def test_no_invalid_affirming_consequent(self):
        result = reason("Если а, то б; б")
        self.assertNotIn("а", result.conclusions)
        self.assertEqual(len(result.steps), 0)

    def test_no_invalid_denial_of_antecedent(self):
        result = reason("Если а, то б; не а")
        self.assertNotIn("не б", result.conclusions)

    def test_verbatim_matching_only_no_synonyms(self):
        result = reason("Если человек свободен, то он выбирает; человек независим")
        self.assertNotIn("он выбирает", result.conclusions)

    def test_syntax_errors(self):
        for value in ("", "а -> ", "а -> б -> в"):
            with self.subTest(value=value), self.assertRaises(ArgumentSyntaxError):
                reason(value)

    def test_repeated_rules_terminate(self):
        result = reason("а -> б; б -> а; а")
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.conclusions, {"а", "б"})


if __name__ == "__main__":
    unittest.main()
