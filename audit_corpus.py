"""Read-only audit of prepared NATTER documents, not of model weights.

Uses the existing corpus report and scans document text for conspicuous
artifacts that often survive wiki markup extraction. It is *not* a general
quality classifier. All output remains local.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import PurePosixPath

from natter.settings import ROOT


def document_category(path: str) -> str:
    parts = PurePosixPath(path).parts
    try:
        index = parts.index("raw")
    except ValueError:
        return "(unknown)"
    tail = parts[index + 1:]
    return tail[0] if len(tail) >= 2 else "(raw root)"


def classify_line(line: str) -> list[str]:
    """High-precision hints only; a URL isn't automatically a bad sentence."""
    patterns = []
    if "thumb|" in line.lower() or "мини|" in line.lower():
        patterns.append("image_markup")
    if "{{" in line or "}}" in line:
        patterns.append("unexpanded_template")
    if "[[" in line or "]]" in line:
        patterns.append("wiki_link")
    if "<ref" in line.lower() or "</ref>" in line.lower():
        patterns.append("wiki_reference")
    if "__NOTOC__" in line or "__TOC__" in line:
        patterns.append("wiki_magic")
    return patterns


def audit_documents(documents, root=ROOT):
    grouped = defaultdict(lambda: {"files": 0, "chars": 0, "train": 0, "val": 0})
    issues = Counter()
    affected = Counter()
    examples = []
    missing = []
    for doc in documents:
        path = doc["path"]
        category = document_category(path)
        item = grouped[category]
        item["files"] += 1
        item["chars"] += doc["chars"]
        item[doc["split"]] += 1

        full_path = root / path
        if not full_path.is_file():
            missing.append(path)
            continue
        flagged = set()
        with full_path.open("r", encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, 1):
                kinds = classify_line(line)
                if kinds:
                    issues.update(kinds)
                    flagged.update(kinds)
                    if len(examples) < 12:
                        examples.append({
                            "path": path, "line": line_no, "markers": kinds,
                            "preview": line.strip()[:140],
                        })
        affected.update(flagged)

    return {
        "categories": dict(sorted(grouped.items())),
        "marker_occurrences": dict(issues),
        "documents_with_markers": dict(affected),
        "examples": examples,
        "missing_documents": missing[:25],
        "missing_count": len(missing),
    }


def main():
    import json
    from datetime import datetime

    parser = argparse.ArgumentParser(description="NATTER: аудит состава и очистки корпуса (только чтение)")
    parser.add_argument("--save", action="store_true", help="Сохранить локальный отчёт в logs/")
    args = parser.parse_args()

    report_path = ROOT / "data" / "processed" / "corpus_report.json"
    if not report_path.is_file():
        raise SystemExit("Сначала подготовь корпус: python prepare.py")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    analysis = audit_documents(report["documents"])
    analysis["total_documents"] = report["unique_documents"]
    analysis["total_chars"] = report["unique_chars"]
    analysis["exact_duplicates"] = len(report.get("exact_duplicates", []))
    analysis["high_internal_repetition"] = len(report.get("high_internal_repetition", []))

    print(f"Документы: {analysis['total_documents']:,}; символы: {analysis['total_chars']:,}")
    print(f"Точные дубликаты: {analysis['exact_duplicates']:,}; "
          f"внутренние повторы (файлов): {analysis['high_internal_repetition']:,}")
    print("\nСостав корпуса:")
    for name, info in analysis["categories"].items():
        share = 100 * info["chars"] / max(1, analysis["total_chars"])
        print(f"  {name}: {info['files']} файлов, {info['chars']:,} символов "
              f"({share:.1f}%), train {info['train']}, val {info['val']}")
    print("\nМаркеры остаточной разметки (не все означают испорченный документ):")
    for name, count in sorted(analysis["marker_occurrences"].items()):
        print(f"  {name}: {count} строк; в {analysis['documents_with_markers'].get(name, 0)} файлах")
    if not analysis["marker_occurrences"]:
        print("  Явных маркеров не обнаружено (это не гарантирует чистоты).")
    for example in analysis["examples"][:5]:
        print(f"  Пример {example['path']}:{example['line']}: {example['preview']}")
    if analysis["missing_count"]:
        print(f"ВНИМАНИЕ: отсутствует {analysis['missing_count']} исходных файлов")
    print("\nВажно: энциклопедия и философские произведения — разные типы данных.")
    print("Этот отчёт не оценивает смысл, аргументацию и навыки диалога.")

    if args.save:
        out = ROOT / "logs" / f"corpus_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Локальный отчёт:", out)


if __name__ == "__main__":
    main()
