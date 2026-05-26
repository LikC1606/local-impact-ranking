from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def convert_prescience_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
    max_candidates: int = 50,
) -> int:
    records = list(_iter_records(input_path))
    return convert_prescience_records(
        records,
        output_path,
        limit=limit,
        max_candidates=max_candidates,
    )


def convert_prescience_records(
    records: Iterable[dict[str, Any]],
    output_path: str | Path,
    *,
    limit: int | None = None,
    max_candidates: int = 50,
) -> int:
    records_list = list(records)
    by_corpus_id = {
        str(record["corpus_id"]): record
        for record in records_list
        if record.get("corpus_id") is not None
    }
    publication_history_by_author_id = _publication_history_by_author_id(records_list)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for record in records_list:
            if not _is_target(record):
                continue
            paper = _target_to_paper(
                record,
                by_corpus_id,
                publication_history_by_author_id,
                max_candidates=max_candidates,
            )
            handle.write(json.dumps(paper, ensure_ascii=False) + "\n")
            count += 1
            if limit is not None and count >= limit:
                break
    return count


def _iter_records(input_path: str | Path) -> Iterable[dict[str, Any]]:
    source = Path(input_path)
    if source.suffix == ".jsonl":
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if stripped:
                    try:
                        yield json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid PreScience JSON on line {line_number}: {exc.msg}"
                        ) from exc
        return
    if source.suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("PreScience JSON input must contain a list of records")
        yield from payload
        return
    if source.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - depends on optional runtime.
            raise RuntimeError("Reading parquet input requires pyarrow") from exc
        parquet = pq.ParquetFile(source)
        for batch in parquet.iter_batches(batch_size=2048):
            yield from batch.to_pylist()
        return
    raise ValueError(f"Unsupported PreScience input format: {source.suffix}")


def _target_to_paper(
    record: dict[str, Any],
    by_corpus_id: dict[str, dict[str, Any]],
    publication_history_by_author_id: dict[str, list[str]],
    *,
    max_candidates: int,
) -> dict[str, Any]:
    positive_ids = _key_reference_ids(record)
    extra_candidate_ids = []
    for author_id in _target_author_ids(record):
        extra_candidate_ids.extend(publication_history_by_author_id.get(author_id, []))
    candidate_ids = _candidate_ids(
        record,
        positive_ids,
        extra_candidate_ids=extra_candidate_ids,
        max_candidates=max_candidates,
    )
    return {
        "paper_id": _record_id(record),
        "title": _text(record.get("title")) or f"PreScience target {_record_id(record)}",
        "abstract": _text(record.get("abstract")),
        "year": _year(record),
        "field": _first(record.get("categories")),
        "authors": _authors(record),
        "references": [
            _reference_from_record(
                candidate_id,
                by_corpus_id.get(candidate_id, {"corpus_id": candidate_id}),
            )
            for candidate_id in candidate_ids
        ],
        "metadata": {
            "source_benchmark": "prescience",
            "arxiv_id": _text(record.get("arxiv_id")),
            "roles": record.get("roles") or [],
            "categories": record.get("categories") or [],
            "citation_trajectory": record.get("citation_trajectory") or [],
            "prescience_positive_reference_ids": positive_ids,
            "prescience_key_reference_citations": _key_reference_citations(record),
        },
    }


def _is_target(record: dict[str, Any]) -> bool:
    roles = set(record.get("roles") or [])
    return "target" in roles or bool(record.get("key_references"))


def _key_reference_ids(record: dict[str, Any]) -> list[str]:
    ids = []
    for item in record.get("key_references") or []:
        if isinstance(item, dict) and item.get("corpus_id") is not None:
            ids.append(str(item["corpus_id"]))
    return _dedupe(ids)


def _key_reference_citations(record: dict[str, Any]) -> dict[str, int]:
    citations = {}
    for item in record.get("key_references") or []:
        if not isinstance(item, dict) or item.get("corpus_id") is None:
            continue
        try:
            citations[str(item["corpus_id"])] = int(item.get("num_citations") or 0)
        except (TypeError, ValueError):
            citations[str(item["corpus_id"])] = 0
    return citations


def _candidate_ids(
    record: dict[str, Any],
    positive_ids: list[str],
    *,
    extra_candidate_ids: list[str],
    max_candidates: int,
) -> list[str]:
    candidate_ids = list(positive_ids)
    target_id = _record_id(record)
    for author in record.get("authors") or []:
        if not isinstance(author, dict):
            continue
        for item in author.get("publication_history") or []:
            if not isinstance(item, dict) or item.get("corpus_id") is None:
                continue
            candidate_id = str(item["corpus_id"])
            if candidate_id != target_id:
                candidate_ids.append(candidate_id)
            if len(_dedupe(candidate_ids)) >= max_candidates:
                return _dedupe(candidate_ids)[:max_candidates]
    for candidate_id in extra_candidate_ids:
        if candidate_id != target_id:
            candidate_ids.append(candidate_id)
        if len(_dedupe(candidate_ids)) >= max_candidates:
            return _dedupe(candidate_ids)[:max_candidates]
    return _dedupe(candidate_ids)[:max_candidates]


def _publication_history_by_author_id(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    history: dict[str, list[str]] = {}
    for record in records:
        if record.get("corpus_id") is None:
            continue
        roles = set(record.get("roles") or [])
        if "target.author.publication_history" not in roles:
            continue
        corpus_id = str(record["corpus_id"])
        for author_id in _target_author_ids(record):
            history.setdefault(author_id, []).append(corpus_id)
    return history


def _target_author_ids(record: dict[str, Any]) -> list[str]:
    ids = []
    for author in record.get("authors") or []:
        if isinstance(author, dict) and author.get("author_id") is not None:
            ids.append(str(author["author_id"]))
    return _dedupe(ids)


def _reference_from_record(reference_id: str, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "reference_id": reference_id,
        "title": _text(record.get("title")) or f"PreScience paper {reference_id}",
        "authors": _authors(record),
        "year": _year(record),
        "field": _first(record.get("categories")),
        "external_ids": {
            key: value
            for key, value in {
                "s2_corpus_id": _text(record.get("corpus_id")) or reference_id,
                "arxiv": _text(record.get("arxiv_id")),
            }.items()
            if value
        },
    }


def _authors(record: dict[str, Any]) -> list[dict[str, Any]]:
    authors = []
    for author in record.get("authors") or []:
        if isinstance(author, dict) and (name := _text(author.get("name"))):
            authors.append({"name": name, "affiliations": []})
    return authors


def _record_id(record: dict[str, Any]) -> str:
    for key in ["corpus_id", "paper_id", "article_id", "arxiv_id"]:
        if record.get(key) is not None:
            return str(record[key])
    raise ValueError("PreScience record is missing corpus_id/arxiv_id")


def _year(record: dict[str, Any]) -> int | None:
    date = _text(record.get("date"))
    if date and len(date) >= 4:
        try:
            return int(date[:4])
        except ValueError:
            return None
    return None


def _first(value: Any) -> str | None:
    if isinstance(value, list) and value:
        return _text(value[0])
    return _text(value)


def _text(value: Any) -> str | None:
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique
