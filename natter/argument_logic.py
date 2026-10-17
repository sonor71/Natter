"""Conservative, deterministic propositional argument checker.

This is a REAL (but narrow) multi-step inference engine, not neural thought.
It only recognizes exactly matching atomic premises and conditional rules.
It does NOT understand philosophical equivalence, synonyms or truth in reality.

Russian notation:
  Если A, то B; Если B, то C; A
  не B; Если A, то B                # modus tollens -> не A

ASCII notation: A -> B; B -> C; A.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

MAX_INPUT_ITEMS = 80
MAX_STEPS = 200


class ArgumentSyntaxError(ValueError):
    pass


def canonical(statement: str) -> str:
    statement = " ".join(statement.strip().rstrip(".!?").split())
    return statement.casefold()


def negate(statement: str) -> str:
    statement = canonical(statement)
    if statement.startswith("не "):
        return statement[3:]
    if statement.startswith("~"):
        return statement[1:]
    return "не " + statement


def parse_item(text: str) -> tuple[str, str, str] | tuple[str, str]:
    text = text.strip()
    if not text:
        raise ArgumentSyntaxError("Пустое утверждение")
    match = re.match(r"^если\s+(.+?),\s*то\s+(.+)$", text, re.I)
    if match:
        antecedent, consequent = canonical(match.group(1)), canonical(match.group(2))
        if not antecedent or not consequent:
            raise ArgumentSyntaxError("Правило должно содержать условие и следствие")
        return ("rule", antecedent, consequent)
    # Explicit arrow is supported in addition to natural language.
    if "->" in text or "→" in text:
        separator = "->" if "->" in text else "→"
        parts = text.split(separator)
        if len(parts) != 2:
            raise ArgumentSyntaxError("Каждое правило должно содержать одну стрелку")
        antecedent, consequent = canonical(parts[0]), canonical(parts[1])
        if not antecedent or not consequent:
            raise ArgumentSyntaxError("Нужно указать обе стороны правила")
        return ("rule", antecedent, consequent)
    result = canonical(text)
    if not result:
        raise ArgumentSyntaxError("Пустой факт")
    return ("fact", result)


@dataclass
class ArgumentResult:
    original: list[str]
    steps: list[tuple[str, str, str, str]]  # derived, rule, support, method
    conclusions: set[str]
    contradictions: list[tuple[str, str]]

    def render(self) -> str:
        lines = [
            "ФОРМАЛЬНЫЙ РАЗБОР (не проверка истинности исходных утверждений):",
            f"Исходных фактов: {len(self.original)}; новых выводов: {len(self.steps)}",
        ]
        if self.steps:
            for index, (derived, rule, support, method) in enumerate(self.steps, 1):
                lines.append(f"{index}. {derived}")
                lines.append(f"   {method}: {rule}; использовано: {support}")
        else:
            lines.append("Новых выводов по этим правилам нет.")
        if self.contradictions:
            lines.append("ПРОТИВОРЕЧИЕ: получены утверждение и отрицание:")
            for a, b in self.contradictions[:5]:
                lines.append(f" - «{a}» и «{b}»")
        else:
            lines.append("Синтаксических противоречий не обнаружено.")
        lines.append("Ограничение: совпадение формулировок должно быть точным; "
                     "смысл, истинность и скрытые посылки не проверяются.")
        return "\n".join(lines)


def reason(text: str) -> ArgumentResult:
    raw_items = [x.strip() for x in text.split(";") if x.strip()]
    if not raw_items:
        raise ArgumentSyntaxError("Пример: Если А, то Б; Если Б, то В; А")
    if len(raw_items) > MAX_INPUT_ITEMS:
        raise ArgumentSyntaxError(f"Слишком много посылок: максимум {MAX_INPUT_ITEMS}")

    rules: list[tuple[str, str]] = []
    known: set[str] = set()
    facts = []
    for item in raw_items:
        parsed = parse_item(item)
        if parsed[0] == "rule":
            rules.append((parsed[1], parsed[2]))
        else:
            fact = parsed[1]
            if fact not in known:
                known.add(fact)
                facts.append(fact)

    steps: list[tuple[str, str, str, str]] = []
    # Propositional modus ponens + modus tollens, no guesswork or semantic jumps.
    # Consequent and antecedent are treated as atomic statements.
    for _ in range(MAX_STEPS):
        additions: list[tuple[str, str, str, str]] = []
        for antecedent, consequent in rules:
            label = f"если {antecedent}, то {consequent}"
            if antecedent in known and consequent not in known:
                additions.append((consequent, label, antecedent, "modus ponens"))
            not_consequent = negate(consequent)
            not_antecedent = negate(antecedent)
            if not_consequent in known and not_antecedent not in known:
                additions.append((not_antecedent, label, not_consequent, "modus tollens"))
        if not additions:
            break
        for derived, label, support, method in additions:
            if derived not in known:
                known.add(derived)
                steps.append((derived, label, support, method))
    else:
        raise RuntimeError("Достигнут защитный предел вывода")

    contradictions = []
    visited = set()
    for statement in sorted(known):
        contrary = negate(statement)
        if contrary in known:
            pair = tuple(sorted((statement, contrary)))
            if pair not in visited:
                visited.add(pair)
                contradictions.append(pair)
    return ArgumentResult(facts, steps, known, contradictions)
