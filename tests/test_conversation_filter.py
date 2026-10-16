"""Prevent regression to math/translation domination in the dialogue corpus."""
import unittest

from import_dialogues import select_pairs
from natter.dialogue_filter import conversational_pair


class ConversationFilterTests(unittest.TestCase):
    def test_arithmetic_and_reverse_text_rejected(self):
        bad = [
            ("893 - 55 =?", "Будет 838. Нужна ещё математика?"),
            ("Привет, посчитай мне 16+2", "Ответ: 18"),
            ("переведи 60 сек в мин", "Получится 1 мин."),
            ("Прочитай слово наоборот", "Задом наперёд — йолс."),
            ("Привет", "Получится 1100."),
            ("как дела?", "Я живу в облаке и не устаю."),
            ("как тебя зовут?", "Я искусственный интеллект, имени нет."),
            ("Переведи слово cat", "Это кот."),
        ]
        for q, a in bad:
            with self.subTest(q=q, a=a):
                self.assertFalse(conversational_pair(q, a))

    def test_simple_conversation_allowed(self):
        good = [
            ("Привет, как настроение?", "Привет! Неплохо, а у тебя как дела?"),
            ("Я сегодня грущу.", "Расскажи, что произошло, если хочешь."),
            ("Поговорим о книгах?", "Давай. Какую книгу ты недавно прочитал?"),
        ]
        for q, a in good:
            with self.subTest(q=q):
                self.assertTrue(conversational_pair(q, a))

    def test_conversation_mode_keeps_answers_disjoint_and_is_deterministic(self):
        # Number-bearing examples are intentionally excluded by the conversational filter.
        # Use unique Cyrillic suffixes instead for the retained examples.
        suffix = lambda i: ''.join(chr(1072 + int(d)) for d in str(i))
        rows = [
            {"q": f"Привет, как настроение сегодня {suffix(i)}!", "a": "Привет! Рад общению."}
            for i in range(20)
        ] + [
            {"q": f"Поговорим о теме {suffix(i)} подробнее?", "a": f"Попробуем обсудить тему {suffix(i)} вместе."}
            for i in range(150)
        ] + [
            {"q": f"Сколько будет {i} + 1?", "a": f"Получится {i+1}."}
            for i in range(100)
        ]
        train, val, stats = select_pairs(
            rows, mode="conversation", max_per_answer=6, seed=9
        )
        self.assertGreater(stats["off_topic"], 90)
        self.assertTrue(all("Получится" not in a for q, a in train + val))
        self.assertTrue(all("Сколько будет" not in q for q, a in train + val))
        self.assertEqual(
            (train, val),
            select_pairs(rows, mode="conversation", max_per_answer=6, seed=9)[:2],
        )
        train_answers = {a for q, a in train}
        val_answers = {a for q, a in val}
        self.assertFalse(train_answers & val_answers)


if __name__ == "__main__":
    unittest.main()
