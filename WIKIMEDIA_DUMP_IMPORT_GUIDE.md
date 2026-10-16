# NATTER CORE — Wikimedia Dump Importer

Расширение к имеющейся сборке, **не заменяет** `train.py`, `prepare.py`, `config.json`, токенизатор или checkpoint.

## 1. В PowerShell

```powershell
cd Y:\NATTER_CORE
.\venv\Scripts\python.exe -m pip install mwparserfromhell
New-Item -ItemType Directory -Force "Y:\NATTER_CORE\downloads"
```

## 2. Скачать одну часть официального архива русской Википедии

Дата архива: latest, Wikimedia. Часть 1 около 398 MB в сжатом виде по состоянию на октябрь 2026.

```powershell
curl.exe --location --retry 3 --continue-at - --output "Y:\NATTER_CORE\downloads\ruwiki_part1.xml.bz2" "https://dumps.wikimedia.org/ruwiki/latest/ruwiki-latest-pages-articles-multistream1.xml-p1p224167.bz2"
```

Никакой облачной модели и API-инференса: только скачивание открытых текстов. Если загрузка не удалась, проверь файл до запуска импортёра:

```powershell
Get-Item "Y:\NATTER_CORE\downloads\ruwiki_part1.xml.bz2" | Select-Object Name, Length
```

## 3. Извлечь ограниченное число очищенных статей

```powershell
.\venv\Scripts\python.exe import_wiki_dump.py --file "downloads/ruwiki_part1.xml.bz2" --max-pages 2000
.\venv\Scripts\python.exe prepare.py --dry-run
```

Можно позднее повторить с `--max-pages 5000`: уже импортированные статьи пропускаются.

Выборка детерминирована по ID страницы; не означает равномерного распределения тем, так как файл представляет собой первую часть дампа.

## 4. Проверка качества

Посмотри случайные документы в `data/raw/wikipedia_dump/`, удали файлы с остаточной разметкой и нерелевантным содержимым. Проект не умеет гарантированно отсеять все ошибки энциклопедии или выявить смысловые дубликаты. Не смешивай тренировочные диалоги с проверочными ответами.

Исходные названия, URL, лицензии и сведения о модификации сохраняются в `data/sources/wikipedia_dump.jsonl`. Wikipedia обычно CC BY-SA 4.0, но нужно соблюдать условия лицензии, включая атрибуцию и распространение изменений на совместимых условиях при применимости.

## 5. Важно при дальнейшей подготовке

До длительного обучения собери значительно больше **разнообразных** качественных текстов и отдельно философские первоисточники. Википедия — полезная основа общего языка и знаний, но сама по себе не обучит Наттера интеллекту Ницше или Батая.

`prepare.py` заново формирует корпус и может пересоздать токенизатор, если его нет. Если модель уже обучалась, нельзя менять BPE при продолжении checkpoint без учёта несовместимости токенов.

Не запускай `train.py` на сотни тысяч шагов до оценки качества и объёма корпуса.
