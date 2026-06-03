from __future__ import annotations

from typing import Any


DANGEROUS_CSV_PREFIXES = ("=", "+", "-", "@")
DANGEROUS_CSV_LEADING_CHARS = (" ", "\t", "\r", "\n")


def sanitize_csv_cell(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value
    candidate = value.lstrip("".join(DANGEROUS_CSV_LEADING_CHARS))
    if candidate.startswith(DANGEROUS_CSV_PREFIXES) or value.startswith(
        ("\t", "\r", "\n")
    ):
        return "'" + value
    return value


def sanitize_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: sanitize_csv_cell(value) for key, value in row.items()}
