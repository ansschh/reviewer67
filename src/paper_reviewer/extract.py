"""PDF -> text extraction. We use pymupdf because OCR'd figures and equations
get lost with most quick-and-dirty extractors.

For your own paper, you'll want the cleanest possible text — equations
especially. If quality matters more than speed, swap in `marker-pdf`."""
from __future__ import annotations

from pathlib import Path

from .config import PATHS


def extract_text(pdf_path: Path | str) -> str:
    import fitz  # pymupdf

    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    pages: list[str] = []
    try:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            pages.append(f"\n\n=== Page {i} ===\n{text}")
    finally:
        doc.close()
    return "".join(pages)


def extract_to_cache(pdf_path: Path | str) -> Path:
    """Extract to data/text/<stem>.txt and return the path."""
    PATHS.ensure()
    pdf_path = Path(pdf_path)
    out = PATHS.text / f"{pdf_path.stem}.txt"
    if not out.exists() or out.stat().st_mtime < pdf_path.stat().st_mtime:
        out.write_text(extract_text(pdf_path), encoding="utf-8")
    return out


def download_pdf(forum_id: str, max_retries: int = 6) -> Path:
    """Download an OpenReview PDF by forum id. Cached on disk.

    Honors HTTP 429 / 503 with exponential backoff. OpenReview's PDF endpoint
    is aggressively rate-limited; without backoff a calibration run loses
    ~10-15% of its sample to skipped PDFs.
    """
    import time
    import urllib.error
    import urllib.request

    PATHS.ensure()
    out = PATHS.pdfs / f"{forum_id}.pdf"
    if out.exists() and out.stat().st_size > 0:
        return out
    url = f"https://openreview.net/pdf?id={forum_id}"
    backoff = 2.0
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "paper-reviewer/0.1"})
            with urllib.request.urlopen(req, timeout=60) as r, out.open("wb") as f:
                while chunk := r.read(64 * 1024):
                    f.write(chunk)
            return out
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt + 1 < max_retries:
                # Honor Retry-After if provided, else exponential backoff.
                wait = float(e.headers.get("Retry-After") or backoff)
                time.sleep(min(wait, 60))
                backoff *= 2
                continue
            raise
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if attempt + 1 < max_retries:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise
    raise RuntimeError(f"download_pdf({forum_id}) exhausted retries")
