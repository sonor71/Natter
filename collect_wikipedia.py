"""Collect Russian Wikipedia plaintext for a from-scratch NATTER corpus.

No model files are downloaded. Uses MediaWiki Action API via Python stdlib.
Saves attribution metadata separately in data/sources/wikipedia.jsonl.
Usage:
    python collect_wikipedia.py --mode philosophy
    python collect_wikipedia.py --mode random --count 100

The Wikimedia API is an external text source, not a pretrained model/API for inference.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
API = "https://ru.wikipedia.org/w/api.php"

TOPICS = [
    "Философия", "Онтология", "Эпистемология", "Метафизика", "Этика",
    "Логика", "Герменевтика", "Феноменология", "Экзистенциализм",
    "Нигилизм", "Мораль", "Свобода воли", "Детерминизм", "Сознание",
    "Философия сознания", "Философия языка", "Проблема зла", "Эстетика",
    "Постструктурализм", "Постмодернизм", "Немецкая классическая философия",
    "Фридрих Ницше", "Жорж Батай", "Иммануил Кант", "Георг Гегель",
    "Артур Шопенгауэр", "Альбер Камю", "Мартин Хайдеггер", "Жан-Поль Сартр",
    "Сверхчеловек", "Воля к власти", "Вечное возвращение", "Генеалогия морали",
    "По ту сторону добра и зла", "Так говорил Заратустра", "Жак Деррида",
    "Мишель Фуко", "Фридрих Шеллинг", "Карл Ясперс", "Жиль Делёз",
]


def parse_retry_after(value: str | None) -> float | None:
    """Parse Retry-After as seconds or as an HTTP date."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        from email.utils import parsedate_to_datetime
        try:
            parsed = parsedate_to_datetime(value)
            return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return None


def request_json(params: dict, user_agent: str, attempts: int = 3) -> dict:
    query = urllib.parse.urlencode({**params, "format": "json", "formatversion": 2, "maxlag": 3})
    req = urllib.request.Request(API + "?" + query, headers={
        "User-Agent": user_agent, "Accept": "application/json",
    })
    for trial in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=50) as response:
                result = json.load(response)
            if "error" in result:
                error = result["error"]
                code = error.get("code", "unknown")
                if code in {"maxlag", "ratelimited"}:
                    if trial == attempts - 1:
                        raise RuntimeError(f"Википедия временно ограничивает запросы ({code}). Попробуй позже.")
                    wait = min(120.0, 15.0 * (2 ** trial))
                    print(f"API: {code}. Пауза {wait:.0f} с...", flush=True)
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Ошибка API: {error}")
            return result
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503):
                wait = parse_retry_after(exc.headers.get("Retry-After"))
                if wait is None:
                    wait = 30.0 * (2 ** trial)
                # A very long Retry-After should not block PowerShell for hours.
                if trial == attempts - 1 or wait > 120:
                    detail = (f"Повторить не раньше чем через {wait:.0f} с."
                              if wait else "Повтори позднее.")
                    raise RuntimeError(
                        f"Wikimedia вернула HTTP {exc.code}. {detail} "
                        "Не запускай сборщик многократно. Проверь --contact."
                    ) from exc
                print(f"HTTP {exc.code}; пауза {wait:.0f} с по правилам Wikimedia...", flush=True)
                time.sleep(wait)
                continue
            raise RuntimeError(f"HTTP {exc.code}: {exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if trial == attempts - 1:
                raise RuntimeError(f"Ошибка соединения с Википедией: {exc}") from exc
            wait = min(60, 5 * 2 ** trial)
            print(f"Ошибка соединения; повтор через {wait} с...", flush=True)
            time.sleep(wait)
    raise RuntimeError("Не удалось получить ответ Wikimedia API")


def normalize_text(value: str) -> str:
    value = html.unescape(value or "").replace("\x00", "")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{4,}", "\n\n\n", value)
    return value.strip()


def pages_from_reply(payload: dict) -> list[dict]:
    return payload.get("query", {}).get("pages", [])


def store_pages(pages: list[dict], mode: str, min_chars: int,
                max_chars: int, existing: set[str], metadata: Path) -> tuple[int, int]:
    added, skipped = 0, 0
    target = ROOT / "data" / "raw" / ("philosophy" if mode == "philosophy" else "encyclopedia")
    target.mkdir(parents=True, exist_ok=True)
    metadata.parent.mkdir(parents=True, exist_ok=True)

    with metadata.open("a", encoding="utf-8") as journal:
        for page in pages:
            page_id = page.get("pageid")
            if page_id is None or "missing" in page or "invalid" in page:
                skipped += 1
                continue
            identifier = str(page_id)
            if identifier in existing:
                skipped += 1
                continue
            title = (page.get("title") or "").strip()
            body = normalize_text(page.get("extract") or "")
            if len(body) < min_chars:
                skipped += 1
                continue
            if max_chars > 0:
                body = body[:max_chars].rstrip()
            # Preserve the title once, but keep copyright attribution in metadata, not repeated boilerplate.
            content = f"{title}\n\n{body}\n"
            out = target / f"wikipedia_{identifier}.txt"
            if out.exists():
                existing.add(identifier)
                skipped += 1
                continue
            out.write_text(content, encoding="utf-8")
            revisions = page.get("revisions", [])
            rev = revisions[0].get("revid") if revisions else None
            record = {
                "source": "Русская Википедия",
                "pageid": page_id,
                "revision_id": rev,
                "title": title,
                "url": f"https://ru.wikipedia.org/?curid={page_id}" + (f"&oldid={rev}" if rev else ""),
                "license_note": "Текст Википедии: CC BY-SA; см. условия страницы и истории правок",
                "collected_utc": datetime.now(timezone.utc).isoformat(),
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "relative_path": str(out.relative_to(ROOT)).replace("\\", "/"),
                "chars": len(content),
            }
            journal.write(json.dumps(record, ensure_ascii=False) + "\n")
            journal.flush()
            existing.add(identifier)
            added += 1
    return added, skipped


def existing_ids(metadata: Path) -> set[str]:
    ids = set()
    if metadata.exists():
        for line in metadata.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    ids.add(str(json.loads(line)["pageid"]))
                except (KeyError, ValueError):
                    pass
    # Also detect manually copied files lacking metadata (do not overwrite).
    for path in (ROOT / "data" / "raw").rglob("wikipedia_*.txt"):
        ids.add(path.stem.removeprefix("wikipedia_"))
    return ids


def fetch_topics(user_agent: str) -> list[dict]:
    pages = []
    for start in range(0, len(TOPICS), 10):
        reply = request_json({
            "action": "query", "titles": "|".join(TOPICS[start:start+10]),
            "prop": "extracts|revisions", "explaintext": "1", "rvprop": "ids",
            "redirects": "1",
        }, user_agent)
        pages.extend(pages_from_reply(reply))
    return pages


def fetch_random(user_agent: str, count: int) -> list[dict]:
    reply = request_json({
        "action": "query", "generator": "random", "grnnamespace": 0,
        "grnlimit": min(count, 10),
        "prop": "extracts|revisions", "explaintext": "1", "rvprop": "ids",
    }, user_agent)
    return pages_from_reply(reply)


def main() -> int:
    ap = argparse.ArgumentParser(description="Собрать русскоязычный корпус из Википедии (без чужих моделей)")
    ap.add_argument("--mode", choices=["philosophy", "random"], required=True)
    ap.add_argument("--count", type=int, default=100,
                    help="Для random: сколько новых документов попытаться сохранить")
    ap.add_argument("--min-chars", type=int, default=500)
    ap.add_argument("--max-chars", type=int, default=25000,
                    help="Максимум символов на документ; 0 = без ограничения")
    ap.add_argument("--delay", type=float, default=6.0,
                    help="Пауза между запросами в секундах; не снижайте без необходимости")
    ap.add_argument("--contact", type=str, default="",
                    help="Обязательно: рабочий email или полный URL страницы участника Wikimedia")
    ap.add_argument("--max-requests", type=int, default=0,
                    help="Ограничить число API-запросов (для проверки поставь 1)")
    args = ap.parse_args()
    if args.count < 1 or args.count > 100000 or args.min_chars < 1 or args.delay < 2.0:
        ap.error("count от 1 до 100000; min-chars >0; delay >=2.0")
    if args.max_requests < 0:
        ap.error("max-requests должен быть >= 0")
    contact = args.contact.strip()
    if not contact or not ("@" in contact or contact.startswith(("https://", "http://"))):
        ap.error('Wikimedia требует контакт в User-Agent: добавь --contact "email@domain.tld" или --contact "https://..."')
    if any(c in contact for c in "\r\n\t"):
        ap.error("Некорректный --contact")
    agent = f"NatterCoreCorpus/1.3 ({contact[:160]}) Python-urllib"
    metadata = ROOT / "data" / "sources" / "wikipedia.jsonl"
    existing = existing_ids(metadata)
    received = skipped = 0
    print("Wikimedia API: запросы последовательные, пауза", args.delay, "с", flush=True)
    if args.mode == "philosophy":
        for n_request, i in enumerate(range(0, len(TOPICS), 10)):
            if args.max_requests and n_request >= args.max_requests:
                break
            payload = request_json({
                "action": "query", "titles": "|".join(TOPICS[i:i+10]),
                "prop": "extracts|revisions", "explaintext": "1", "rvprop": "ids",
                "redirects": "1",
            }, agent)
            added, dropped = store_pages(pages_from_reply(payload), args.mode, args.min_chars,
                                         args.max_chars, existing, metadata)
            received += added
            skipped += dropped
            print(f"Философские статьи: сохранено {received}; пропущено {skipped}", flush=True)
            if i + 10 < len(TOPICS):
                time.sleep(args.delay)
    else:
        tries = 0
        max_tries = max(30, (args.count // 10 + 1) * 10)
        while received < args.count and tries < max_tries:
            if args.max_requests and tries >= args.max_requests:
                break
            tries += 1
            pages = fetch_random(agent, min(10, args.count - received))
            added, dropped = store_pages(pages, args.mode, args.min_chars,
                                         args.max_chars, existing, metadata)
            received += added
            skipped += dropped
            print(f"Сохранено: {received}/{args.count}; пропущено: {skipped}", flush=True)
            if received < args.count:
                time.sleep(args.delay)
        if received < args.count:
            print(f"Удалось сохранить только {received} статей: часть оказалась короткой или недоступной.")
    print(f"Готово. Новые документы: {received}. Источники: {metadata.relative_to(ROOT)}")
    print("Проверь корпус командой: .\\venv\\Scripts\\python.exe prepare.py --dry-run")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, urllib.error.URLError) as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        raise SystemExit(1)
