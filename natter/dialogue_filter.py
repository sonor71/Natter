"""Conservative rule-based subset for early short conversational fine-tuning.

The source dataset has many arithmetic and transformation tasks with unique
answers, while repeated greetings are capped. Naive answer deduplication
therefore over-represents arithmetic. This filter is deliberately *not* a
semantic classifier or quality guarantee.
"""
from __future__ import annotations

import re

NUMERIC_OR_FORMULA = re.compile(r"\d|[=×÷+%²³]|\b(?:плюс|минус|умнож|делён|делить|вычти|сложи|вычисл|посчитай|процент|число|арифметик)\w*", re.I)
TRANSLATION = re.compile(
    r"\b(?:переведи|переводится|перевод|по-английски|по-русски|"
    r"как будет .* на английском|что значит .* на английском|слово .* по-английски)\b",
    re.I,
)
REVERSE_WORD = re.compile(r"\b(?:наоборот|задом наперёд|справа налево|переверни слово)\b", re.I)
OTHER_IDENTITY = re.compile(
    r"\b(?:я живу в облаке|я родился в момент запуска|"
    r"меня создал(?:и)?|я искусственный интеллект|я ии|"
    r"я программа|как chatgpt|как ии|"
    r"я иИ-собеседник|у меня нет имени|зови меня как хочешь)\b",
    re.I,
)
MATH_ANSWER = re.compile(
    r"^(?:получится|будет|ответ\s*:|если не ошибаюсь,\s*\d)|"
    r"\b(?:нужна ещё математика|переводится как|по-русски это|по-английски это)\b",
    re.I,
)


def conversational_pair(q: str, answer: str) -> bool:
    """Heuristic filter focused on greetings, personal conversation and advice."""
    if not (3 <= len(q) <= 180 and 3 <= len(answer) <= 350):
        return False
    for value in (q, answer):
        if NUMERIC_OR_FORMULA.search(value):
            return False
        if TRANSLATION.search(value) or REVERSE_WORD.search(value):
            return False
        if re.search(r"https?://|\{\{|\[\[|\x00", value):
            return False
    if MATH_ANSWER.search(answer) or OTHER_IDENTITY.search(answer):
        return False
    if any(k in answer.casefold() for k in (
        "я живу в облаке", "как ии", "я иИ".casefold(), "в момент запуска модели",
    )):
        return False
    return True
