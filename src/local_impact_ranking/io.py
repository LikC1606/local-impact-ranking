from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from local_impact_ranking.models import Paper


def load_papers_jsonl(path: str | Path) -> list[Paper]:
    papers: list[Paper] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
            try:
                papers.append(Paper.model_validate(record))
            except ValidationError as exc:
                raise ValueError(f"Invalid paper record on line {line_number}: {exc}") from exc
    return papers


def convert_papers_csv_to_jsonl(input_path: str | Path, output_path: str | Path) -> int:
    source = Path(input_path)
    destination = Path(output_path)
    count = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", newline="", encoding="utf-8") as input_handle, destination.open(
        "w",
        encoding="utf-8",
    ) as output_handle:
        reader = csv.DictReader(input_handle)
        for row_number, row in enumerate(reader, start=2):
            record = _paper_record_from_csv_row(row, row_number)
            try:
                paper = Paper.model_validate(record)
            except ValidationError as exc:
                raise ValueError(f"Invalid CSV paper record on row {row_number}: {exc}") from exc
            output_handle.write(json.dumps(paper.model_dump(), ensure_ascii=False) + "\n")
            count += 1
    return count


def _paper_record_from_csv_row(row: dict[str, str | None], row_number: int) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key in ["paper_id", "title", "abstract", "venue", "field"]:
        value = _optional_cell(row.get(key))
        if value is not None:
            record[key] = value
    year = _optional_cell(row.get("year"))
    if year is not None:
        try:
            record["year"] = int(year)
        except ValueError as exc:
            raise ValueError(f"Invalid year on row {row_number}: {year}") from exc
    for column, output_key in [
        ("authors_json", "authors"),
        ("references_json", "references"),
        ("citation_contexts_json", "citation_contexts"),
        ("metadata_json", "metadata"),
    ]:
        value = _optional_cell(row.get(column))
        if value is not None:
            record[output_key] = _parse_json_cell(value, column, row_number)
    return record


def _optional_cell(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _parse_json_cell(value: str, column: str, row_number: int) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in column {column} on row {row_number}: {exc.msg}"
        ) from exc
