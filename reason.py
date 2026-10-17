"""Offline multi-step propositional inference, no neural model required.

Example:
  python reason.py --text "Если а, то б; Если б, то в; а"
Or run as a prompt and enter statements separated by semicolons.
"""
from __future__ import annotations

import argparse

from natter.argument_logic import ArgumentSyntaxError, reason


def main():
    parser = argparse.ArgumentParser(
        description="Строгая формальная логика NATTER (modus ponens/tollens)"
    )
    parser.add_argument("--text", help="Список правил и посылок через точку с запятой")
    args = parser.parse_args()
    if args.text:
        try:
            print(reason(args.text).render())
        except ArgumentSyntaxError as exc:
            parser.error(str(exc))
        return
    print("Пиши факты и правила через ';'. Выход: /exit")
    while True:
        try:
            value = input("Посылки: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nВыход")
            break
        if value in ("/exit", "выход"):
            break
        try:
            print(reason(value).render())
        except ArgumentSyntaxError as exc:
            print("Ошибка:", exc)


if __name__ == "__main__":
    main()
