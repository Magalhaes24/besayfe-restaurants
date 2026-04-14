from __future__ import annotations

import csv
import io


async def extract_csv(file_bytes: bytes) -> dict:
    """
    Detect encoding, parse CSV, return headers + rows + raw text for AI.
    Restaurants export CSVs with cp1252, iso-8859-1, or utf-8 — detect all.
    """
    from charset_normalizer import from_bytes

    # Encoding detection
    best = from_bytes(file_bytes).best()
    encoding = best.encoding if best else "utf-8"

    try:
        text = file_bytes.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        text = file_bytes.decode("latin-1", errors="replace")

    # Try to detect delimiter (comma, semicolon, tab are all common)
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = "excel"  # fallback to excel dialect

    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    rows = [dict(row) for row in reader]
    headers = list(rows[0].keys()) if rows else []

    return {
        "headers": headers,
        "rows": rows,
        "raw_text": text,
        "row_count": len(rows),
        "encoding_detected": encoding,
    }
