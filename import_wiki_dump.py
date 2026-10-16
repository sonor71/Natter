"""Offline Wikimedia XML .bz2 importer with conservative wikitext cleaning.

- Streams XML pages, avoiding loading the complete dump into memory.
- Renders link labels, useful language-template parameters, and headings.
- Drops image/category links, references, non-prose markup, and bibliography tails.
- --rebuild stages a NEW corpus, then archives and replaces the old import.
- Does not modify the tokenizer, checkpoints, or other data/raw categories.

Examples (run from NATTER_CORE directory):
  python import_wiki_dump.py --file downloads/ruwiki_part1.xml.bz2 --max-pages 10000 --dry-run --rebuild
  python import_wiki_dump.py --file downloads/ruwiki_part1.xml.bz2 --max-pages 10000 --rebuild
"""
from __future__ import annotations

import argparse
import bz2
import hashlib
import html
import json
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_NAME = "Russian Wikipedia"
LICENSE = "Wikipedia text, generally CC BY-SA 4.0; check individual page and media exceptions"

# Ignore sections that are primarily bibliographies / references, not prose.
BACKMATTER_RE = re.compile(
    r"(?im)^\s*={2,6}\s*(?:Примечания|Ссылки|Литература|Библиография|"
    r"См\.?\s*также|Источники|Сноски|Комментарии|Галерея)\s*={2,6}\s*$"
)
MEDIA_PREFIXES = (
    "файл:", "file:", "изображение:", "image:", "media:",
    "категория:", "category:", "special:", "специальная:",
)
REMOVE_TAGS = {
    "ref", "references", "gallery", "math", "timeline", "graph", "mapframe",
    "maplink", "syntaxhighlight", "source", "code", "pre", "score", "templatestyles",
    "noinclude", "includeonly", "nowiki", "poem", "indicator",
}
INLINE_BREAK_TAGS = {"br", "hr", "li", "tr", "p", "div"}
EMPTY_PARENS_RE = re.compile(r"\(\s*\)")
RESIDUAL_MARKUP_RE = re.compile(
    r"\{\{|\}\}|\[\[|\]\]|</?ref\b|(?:^|\s)(?:thumb|мини|миниатюра)\s*\|",
    flags=re.IGNORECASE,
)
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")


def child_text(node: ET.Element, name: str) -> str:
    return (node.findtext(f"{{*}}{name}") or "").strip()


def extract(page: ET.Element):
    if child_text(page, "ns") != "0" or page.find("{*}redirect") is not None:
        return None
    page_id, title = child_text(page, "id"), child_text(page, "title")
    text = page.find("{*}revision/{*}text")
    if not page_id.isdigit() or not title or text is None or not text.text:
        return None
    return page_id, title, text.text


def iter_pages(file_path: Path):
    with bz2.open(file_path, "rb") as fh:
        iterator = ET.iterparse(fh, events=("start", "end"))
        _, root = next(iterator)
        for event, element in iterator:
            if event == "end" and element.tag.rsplit("}", 1)[-1] == "page":
                yield element
                root.clear()


def _render(code, mw, depth=0) -> str:
    """Render Wikicode recursively without leaking File/Category or ref contents."""
    if code is None or depth > 20:
        return ""
    if not hasattr(code, "nodes"):
        code = mw.parse(str(code))
    parts = []
    for node in code.nodes:
        kind = type(node).__name__
        if kind == "Text":
            parts.append(str(node))
        elif kind == "Wikilink":
            title = str(node.title).strip().replace("_", " ")
            if title.lstrip(":").lower().startswith(MEDIA_PREFIXES):
                parts.append(" ")
            else:
                parts.append(_render(node.text if node.text is not None else node.title, mw, depth + 1))
        elif kind == "ExternalLink":
            # Bare external URLs add little linguistic value; labelled links add their caption.
            parts.append(_render(node.title, mw, depth + 1) if node.title is not None else " ")
        elif kind == "Template":
            name = re.sub(r"[\s_]+", " ", str(node.name)).strip().lower()
            param = None
            if name.startswith(("lang-", "lang ")) or name in {"nowrap", "nobr", "неразр", "abbr", "аббр", "quote", "цитата", "цитата текст", "форматчисла"}:
                param = 1
            elif name in {"lang", "язык", "langx"}:
                param = 2
            if param is not None and node.has(param):
                parts.append(" " + _render(node.get(param).value, mw, depth + 1) + " ")
            else:
                # Unknown templates (infoboxes, citations, navigation) are dropped.
                parts.append(" ")
        elif kind == "Tag":
            tagname = str(node.tag).strip().lower()
            if tagname in REMOVE_TAGS:
                parts.append(" ")
            elif tagname in INLINE_BREAK_TAGS:
                inner = _render(node.contents, mw, depth + 1)
                parts.append("\n" + inner + "\n")
            else:
                parts.append(_render(node.contents, mw, depth + 1))
        elif kind == "Heading":
            parts.append("\n" + _render(node.title, mw, depth + 1).strip() + "\n")
        elif kind == "HTMLEntity":
            parts.append(html.unescape(str(node)))
        elif kind == "Argument":
            parts.append(_render(node.default, mw, depth + 1) if node.default is not None else " ")
        elif kind == "Comment":
            parts.append(" ")
        else:
            # Intentionally do not copy raw unknown syntax into training text.
            parts.append(" ")
    return "".join(parts)


def normalize_text(text: str) -> str:
    """Conservative typography cleanup, keeping paragraph/sentence boundaries."""
    text = html.unescape(text).replace("\xa0", " ").replace("\u200b", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?m)^\s*__[^\n_]+__\s*$", "", text)
    text = re.sub(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", "", text)
    text = EMPTY_PARENS_RE.sub("", text)
    # Wikitext bold/italic apostrophes are plain text nodes in many parsers.
    text = re.sub(r"(?<!')'{2,5}(?!')", "", text)
    cleaned = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t\f\v]+", " ", line).strip()
        # Known image parameter fragments from malformed markup.
        if re.match(r"(?i)^(?:thumb|мини|миниатюра)\s*\|", line):
            continue
        if re.match(r"(?i)^https?://\S+\s*$", line):
            continue
        cleaned.append(line)
    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" +([.,;:!?])", r"\1", text)
    return text.strip()


def clean_wikitext(raw: str, mw) -> str:
    found = BACKMATTER_RE.search(raw)
    if found and found.start() > 300:
        raw = raw[:found.start()]
    return normalize_text(_render(mw.parse(raw), mw))


def quality_issues(text: str, min_chars: int, min_cyrillic_ratio: float) -> list[str]:
    if len(text) < min_chars:
        return ["too_short"]
    ratio = len(CYRILLIC_RE.findall(text)) / max(1, len(text))
    issues = []
    if ratio < min_cyrillic_ratio:
        issues.append("low_cyrillic_ratio")
    if RESIDUAL_MARKUP_RE.search(text):
        issues.append("residual_markup")
    # Penalize texts that consist mostly of short headings/lists rather than prose.
    substantial_lines = [s for s in text.splitlines() if len(s.strip()) >= 80]
    if len(text) >= 2000 and not substantial_lines:
        issues.append("no_prose")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Чистый локальный импорт русской Википедии из .xml.bz2")
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--max-pages", type=int, default=2000,
                        help="Сколько пригодных статей создать за запуск")
    parser.add_argument("--min-chars", type=int, default=700)
    parser.add_argument("--max-chars-per-page", type=int, default=25000)
    parser.add_argument("--min-cyrillic-ratio", type=float, default=0.22)
    parser.add_argument("--sample-mod", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rebuild", action="store_true",
                        help="Пересобрать весь импорт во временной папке; старое сохранить в archive/")
    args = parser.parse_args()
    if (args.max_pages < 1 or args.min_chars < 120 or args.sample_mod < 1
        or args.max_chars_per_page < args.min_chars
        or not 0 <= args.min_cyrillic_ratio <= 1):
        parser.error("Недопустимые настройки импорта")
    dump = args.file if args.file.is_absolute() else ROOT / args.file
    if not dump.is_file():
        parser.error(f"Архив не найден: {dump}")
    try:
        import mwparserfromhell as mw
    except ImportError as exc:
        raise SystemExit(
            "Нужен mwparserfromhell. Установка: "
            r".\venv\Scripts\python.exe -m pip install mwparserfromhell"
        ) from exc

    old_target = ROOT / "data" / "raw" / "wikipedia_dump"
    old_journal = ROOT / "data" / "sources" / "wikipedia_dump.jsonl"
    staging_root = None
    if args.rebuild and not args.dry_run:
        staging_root = Path(tempfile.mkdtemp(prefix="natter_wiki_", dir=ROOT))
        target = staging_root / "wikipedia_dump"
        journal_path = staging_root / "wikipedia_dump.jsonl"
    else:
        target, journal_path = old_target, old_journal

    if not args.dry_run:
        target.mkdir(parents=True, exist_ok=True)
        journal_path.parent.mkdir(parents=True, exist_ok=True)

    extracted = visited = skipped = old = characters = 0
    reasons: dict[str, int] = {}
    journal = None
    completed = False
    try:
        if not args.dry_run:
            journal = journal_path.open("w" if args.rebuild else "a", encoding="utf-8")
        for page in iter_pages(dump):
            visited += 1
            candidate = extract(page)
            if candidate is None:
                continue
            page_id, title, raw = candidate
            if int(hashlib.sha256(page_id.encode("ascii")).hexdigest()[:8], 16) % args.sample_mod:
                continue
            filename = target / f"ruwiki_{page_id}.txt"
            if not args.rebuild and filename.exists():
                old += 1
                continue
            text = clean_wikitext(raw, mw)
            if len(text) > args.max_chars_per_page:
                # Avoid severing a sentence in the middle when possible.
                cut = text[:args.max_chars_per_page]
                end = max(cut.rfind(". "), cut.rfind(".\n"), cut.rfind("! "), cut.rfind("? "))
                text = cut[:end + 1] if end >= args.min_chars else cut.rsplit("\n", 1)[0] or cut
            problems = quality_issues(text, args.min_chars, args.min_cyrillic_ratio)
            if problems:
                skipped += 1
                for reason in problems:
                    reasons[reason] = reasons.get(reason, 0) + 1
                continue
            content = f"{title}\n\n{text}\n"
            if not args.dry_run:
                filename.write_text(content, encoding="utf-8")
                journal.write(json.dumps({
                    "path": str((old_target / filename.name).relative_to(ROOT)).replace("\\", "/"),
                    "title": title,
                    "page_id": page_id,
                    "source": SOURCE_NAME,
                    "url": f"https://ru.wikipedia.org/?curid={page_id}",
                    "license": LICENSE,
                    "dump": dump.name,
                    "modification": "Wikitext cleaned with custom AST renderer; bibliographies removed; possible truncation",
                }, ensure_ascii=False) + "\n")
            extracted += 1
            characters += len(content)
            if extracted % 250 == 0:
                print(f"Статей: {extracted:,}; символов: {characters:,}; просмотрено: {visited:,}", flush=True)
                if journal:
                    journal.flush()
            if extracted >= args.max_pages:
                break
        completed = True
    except (OSError, ET.ParseError) as exc:
        raise SystemExit(f"Ошибка чтения или записи архива: {exc}") from exc
    finally:
        if journal:
            journal.close()

    if args.rebuild and not args.dry_run and completed:
        if not extracted:
            raise SystemExit(f"Ни одной пригодной статьи; исходный корпус не затронут. Временная папка: {staging_root}")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = ROOT / "archive" / f"wiki_before_cleanup_{stamp}"
        if backup_dir.exists():
            backup_dir = ROOT / "archive" / f"wiki_before_cleanup_{stamp}_{stamp[-3:]}"
        backup_dir.mkdir(parents=True)
        if old_target.exists():
            shutil.move(str(old_target), str(backup_dir / "wikipedia_dump"))
        if old_journal.exists():
            shutil.move(str(old_journal), str(backup_dir / "wikipedia_dump.jsonl"))
        old_target.parent.mkdir(parents=True, exist_ok=True)
        old_journal.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target), str(old_target))
        shutil.move(str(journal_path), str(old_journal))
        staging_root.rmdir()
        print(f"Резервная копия старого импорта: {backup_dir}")
        print("Токенизатор, .bin и checkpoints НЕ изменялись.")

    print(f"Готово: статей {extracted:,}; символов {characters:,}; просмотрено {visited:,}; "
          f"отклонено {skipped:,}; уже было {old:,}.")
    if reasons:
        print("Причины отклонения:", ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())))
    if args.dry_run:
        print("DRY RUN: никакие файлы не изменены.")
    else:
        print(f"Тексты: {old_target}")
        print(f"Каталог источников: {old_journal}")
        print("Выборочная ручная проверка текстов всё ещё необходима.")


if __name__ == "__main__":
    main()
