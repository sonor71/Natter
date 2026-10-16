NATTER Wikimedia API 429 fix

Updated collector:
- Requires a meaningful contact in the Wikimedia User-Agent header
- Honors Retry-After for HTTP 429/503
- Uses bounded backoff and stops if Wikimedia asks for a long wait
- Default 6-second pause between requests
- --max-requests 1 allows a single-request test

Usage in Windows PowerShell:
cd Y:\NATTER_CORE
.\venv\Scripts\python.exe collect_wikipedia.py --mode philosophy --contact "YOUR_REAL_CONTACT" --max-requests 1

For YOUR_REAL_CONTACT, use your own email or a full URL to your Wikimedia user page.
Do not use a fictitious contact or attempt to bypass limits.

Run longer only after the initial request succeeds, and only within rate limits.
The Wikimedia API is not the best option for bulk corpora: see https://dumps.wikimedia.org/ for dumps.
