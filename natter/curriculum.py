"""Small, fully local reasoning-and-context curriculum for NATTER.

These examples are *synthetic exercises*, not philosophical primary sources.
The generation rules are transparent and train/val scenarios never overlap.
Template reuse limits generalization; a good validation loss does not establish
independent logical understanding.

No third-party pretrained models, APIs or online service are used.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from .philosophy_cases import examples as philosophy_examples

# Each scenario is an explicit assumption chain A -> B -> C, not a claim
# that either conditional is necessarily true in the real world.
# name is for the separate memory exercise.
SCENARIOS = [
    ("письмо отправлено", "письмо доставлено", "получатель может прочитать письмо", "Север"),
    ("редактор одобрил черновик", "текст принят к выпуску", "выпуск можно планировать", "Янтарь"),
    ("человек сделал осознанный выбор", "у него была возможность выбирать", "можно обсуждать ответственность за решение", "Маяк"),
    ("исследователь зафиксировал гипотезу", "гипотеза записана", "гипотезу можно проверять", "Лабиринт"),
    ("книга поступила в каталог", "у книги есть запись", "читатель может искать её по записи", "Переплёт"),
    ("дверь заперта", "для входа нужен способ открыть дверь", "вход требует дополнительного действия", "Кварц"),
    ("команда согласовала цель", "есть общая формулировка задачи", "можно проверить результат относительно задачи", "Такт"),
    ("участники определили правила спора", "критерии обсуждения ясны", "спор можно оценивать по этим критериям", "Мост"),
    ("автор привёл источник", "утверждение можно сопоставить с источником", "можно проверить точность цитаты", "Архив"),
    ("ученик сформулировал вопрос", "вопрос стал явным", "можно искать относящиеся к нему доводы", "Искра"),
    ("разговор записан", "есть последовательность реплик", "можно восстановить ход обсуждения", "Эхо"),
    ("проект прошёл проверку", "критические ошибки не обнаружены", "проект допустили к испытаниям", "Орбита"),
    ("событие получило дату", "дата указана в записи", "можно сравнивать порядок событий", "Хроника"),
    ("архитектор составил план", "есть описание будущего сооружения", "можно сопоставить стройку с планом", "Контур"),
    ("исследователь определил понятие", "границы понятия заданы", "можно проверять применение определения", "Форма"),
    ("переговоры завершились согласием", "стороны приняли условие", "можно проверять исполнение условия", "Согласие"),
    ("маршрут отмечен на карте", "существует описание пути", "можно обсуждать альтернативные маршруты", "Компас"),
    ("историк нашёл документ", "появилось письменное свидетельство", "версию события можно проверять по свидетельству", "Свиток"),
    ("музыкант записал мелодию", "появилась нотная запись", "другой музыкант может исполнить запись", "Аккорд"),
    ("наблюдатель уточнил критерий", "стало ясно, что измерять", "результаты можно сравнивать по критерию", "Мера"),
    ("читатель заметил противоречие", "два тезиса несовместимы", "нужно пересмотреть хотя бы один тезис", "Диалог"),
    ("наставник объяснил правило", "правило сформулировано", "можно проверить пример по правилу", "Порог"),
    ("исследователь выделил причину", "сформулирована причинная гипотеза", "можно искать опровергающие её наблюдения", "Сфера"),
    ("писатель завершил главу", "есть законченный фрагмент", "редактор может оценить целостность фрагмента", "Лист"),
    # Below: validation-only scenarios. None of their specific propositions
    # or names occur as training curriculum cases.
    ("фотограф сохранил негатив", "появился исходный снимок", "снимок можно сопоставить с копиями", "Штатив"),
    ("астроном уточнил координаты", "положение объекта записано", "другие наблюдения можно сравнить с записью", "Зенит"),
    ("переводчик закончил рукопись", "есть полный вариант перевода", "перевод можно сверить с оригиналом", "Словарь"),
    ("путешественник оставил маршрут", "путь описан в дневнике", "можно восстановить последовательность остановок", "Берег"),
    ("садовник пометил дерево", "у дерева есть отметка", "его можно отличить от соседних деревьев", "Ветка"),
    ("инженер проверил датчик", "известна погрешность измерения", "можно оценивать точность показаний", "Импульс"),
    ("археолог нашёл печать", "появился новый артефакт", "его происхождение можно исследовать", "Раскоп"),
    ("редактор отметил спорный абзац", "замечание записано", "автор может вернуться к спорному месту", "Пометка"),
]


def scenario_split(index: int) -> str:
    return "train" if index < 24 else "val"


def dialogue_cases(a: str, b: str, c: str, name: str):
    """Yield distinct skills with up to three verbal formulations per skill."""
    variants = [
        ("Допустим, что", "Какой вывод следует?", "Покажи цепочку вывода."),
        ("Предположим, что", "Что отсюда можно вывести?", "Назови шаги рассуждения."),
        ("Примем для задачи, что", "Как это связано со следствием?", "Сделай последовательный вывод."),
    ]
    for vi, (opening, query, final_question) in enumerate(variants):
        p1 = f"если {a}, то {b}"
        p2 = f"если {b}, то {c}"
        # The final answer explicitly distinguishes rule assumptions from truth.
        yield {
            "skill": "chain",
            "variant": vi,
            "turns": [
                {"user": f"{opening} {p1}.", "assistant": "Я принимаю это как условное правило задачи, а не как доказанный факт."},
                {"user": f"Добавим второе правило: {p2}.", "assistant": "Теперь есть цепочка из двух условных правил."},
                {"user": f"Известно, что {a}. {final_question}", "assistant": f"Сначала следует, что {b}. Затем следует, что {c}. Это вывод из принятых условий."},
            ],
        }
        yield {
            "skill": "converse",
            "variant": vi,
            "turns": [
                {"user": f"В задаче действует правило: {p1}.", "assistant": "Правило сообщает, что первое условие достаточно для второго."},
                {"user": f"Пока известно только следствие: {b}.", "assistant": "Наличие следствия само по себе ещё не доказывает исходное условие."},
                {"user": f"Можно ли доказать, что {a}?", "assistant": f"Нет. Из правила «{p1}» и факта «{b}» нельзя заключить, что {a}. Это ошибка подтверждения следствия."},
            ],
        }
        yield {
            "skill": "contradiction",
            "variant": vi,
            "turns": [
                {"user": f"Примем правило: {p1}.", "assistant": "Условное правило принято для проверки."},
                {"user": f"При этом известно, что {a}, но неверно, что {b}.", "assistant": "Посылка и отрицание ожидаемого следствия уже противоречат правилу."},
                {"user": f"В чём конкретно противоречие?", "assistant": f"По правилу из «{a}» следует «{b}». Но одновременно утверждается «не {b}». Эти утверждения несовместимы при данном правиле."},
            ],
        }
        yield {
            "skill": "uncertainty",
            "variant": vi,
            "turns": [
                {"user": f"Предположим условие: {p1}.", "assistant": "Это утверждение о связи двух событий, но не о том, что первое произошло."},
                {"user": "Никаких сведений об исполнении первого условия у нас нет.", "assistant": "Тогда я не могу считать следствие установленным фактом."},
                {"user": f"Следует ли утверждать, что {b}?", "assistant": f"Нет. Пока неизвестно, выполняется ли условие «{a}», вывод о том, что {b}, не обоснован."},
            ],
        }
        yield {
            "skill": "memory",
            "variant": vi,
            "turns": [
                {"user": f"Для этого разговора запомни условное название проекта: {name}.", "assistant": f"В текущем разговоре проект назван «{name}»."},
                {"user": "Мне важно, чтобы ты не подменял название догадкой.", "assistant": "Буду опираться на название, которое ты сообщил, а не придумывать другое."},
                {"user": f"{query} Какое название проекта я называл?", "assistant": f"Ты назвал проект «{name}»."},
            ],
        }


def build(seed: int = 1337):
    """Train and val differ by entire scenarios, not by random row."""
    splits = {"train": [], "val": []}
    for i, (a, b, c, name) in enumerate(SCENARIOS):
        split = scenario_split(i)
        for row in dialogue_cases(a, b, c, name):
            splits[split].append({
                "turns": row["turns"],
                "skill": row["skill"],
                "scenario": i,
                "variant": row["variant"],
            })
    # Genuine editorial examples are kept separate from algorithmically varied
    # exercises and are not silently multiplied to inflate dataset size.
    authored = philosophy_examples()
    for name in ('train', 'val'):
        splits[name].extend(authored[name])
    rng = random.Random(seed)
    for records in splits.values():
        rng.shuffle(records)
    return splits


def write_curriculum(root: Path, seed: int = 1337, overwrite: bool = False):
    destination = root / "data" / "dialogues" / "curriculum"
    outputs = [destination / (name + ".jsonl") for name in ("train", "val")]
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError(
            f"{destination} уже содержит файлы. Сохрани предыдущие версии и "
            "используй --force только если сознательно начинаешь новый датасет."
        )
    splits = build(seed)
    destination.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        path = destination / (split + ".jsonl")
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as out:
            for row in splits[split]:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(path)
    metadata = {
        "source": "NATTER transparent synthetic logical exercises (code-generated)",
        "license": "Project-authored examples; not external model output",
        "seed": seed,
        "scenarios_train": 24,
        "scenarios_validation": len(SCENARIOS) - 24,
        "train_examples": len(splits["train"]),
        "validation_examples": len(splits["val"]),
        "author_written_philosophy_train": len(philosophy_examples()["train"]),
        "author_written_philosophy_validation": len(philosophy_examples()["val"]),
        "limitations": "Most examples are template-generated; validation holds out scenarios, not templates. "
                       "The small independently authored philosophy portion holds out topics. "
                       "Neither split alone proves general reasoning or philosophical comprehension.",
    }
    (destination / "SOURCE.json").write_text(json.dumps(
        metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata
