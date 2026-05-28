from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


def convert_sciimpact_jsonl(input_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source = Path(input_path)
    destination = Path(output_dir)
    papers: OrderedDict[str, dict[str, Any]] = OrderedDict()
    labels: list[dict[str, Any]] = []

    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            parsed = _parse_pair(record, line_number)
            winner = parsed["winner"]
            loser = parsed["loser"]
            winner_id = _paper_id(winner, f"line-{line_number}-winner")
            loser_id = _paper_id(loser, f"line-{line_number}-loser")
            pair_id = str(record.get("id") or record.get("pair_id") or f"sciimpact-{line_number}")
            impact_dimension = _first_string(
                record,
                ("impact_dimension", "dimension", "impact_type", "category", "task"),
            )
            field = _first_string(record, ("field", "domain", "discipline", "subject"))

            _upsert_paper(
                papers,
                winner,
                winner_id,
                pair_id=pair_id,
                role="winner",
                impact_dimension=impact_dimension,
                field=field,
            )
            _upsert_paper(
                papers,
                loser,
                loser_id,
                pair_id=pair_id,
                role="loser",
                impact_dimension=impact_dimension,
                field=field,
            )
            labels.append(
                {
                    "pair_id": pair_id,
                    "winner_id": winner_id,
                    "loser_id": loser_id,
                    "impact_dimension": impact_dimension or "unknown",
                    "field": field or "unknown",
                    "source_benchmark": "sciimpact",
                    "source_line": line_number,
                }
            )

    destination.mkdir(parents=True, exist_ok=True)
    papers_path = destination / "papers.jsonl"
    labels_path = destination / "pairwise_labels.jsonl"
    with papers_path.open("w", encoding="utf-8") as handle:
        for paper in papers.values():
            handle.write(json.dumps(paper, ensure_ascii=False) + "\n")
    with labels_path.open("w", encoding="utf-8") as handle:
        for label in labels:
            handle.write(json.dumps(label, ensure_ascii=False) + "\n")

    manifest = {
        "source_benchmark": "sciimpact",
        "source_path": str(source),
        "paper_count": len(papers),
        "pair_count": len(labels),
        "papers_path": str(papers_path),
        "pairwise_labels_path": str(labels_path),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parse_pair(record: dict[str, Any], line_number: int) -> dict[str, dict[str, Any]]:
    if isinstance(record.get("chosen"), dict) and isinstance(record.get("rejected"), dict):
        return {"winner": record["chosen"], "loser": record["rejected"]}
    if isinstance(record.get("positive"), dict) and isinstance(record.get("negative"), dict):
        return {"winner": record["positive"], "loser": record["negative"]}
    if isinstance(record.get("winner"), dict) and isinstance(record.get("loser"), dict):
        return {"winner": record["winner"], "loser": record["loser"]}

    paper_a = record.get("paper_a") or record.get("a")
    paper_b = record.get("paper_b") or record.get("b")
    winner = record.get("winner") or record.get("label") or record.get("answer")
    if isinstance(paper_a, dict) and isinstance(paper_b, dict):
        winner_key = str(winner or "").strip().lower()
        if winner_key in {"paper_a", "a", "0", "left"}:
            return {"winner": paper_a, "loser": paper_b}
        if winner_key in {"paper_b", "b", "1", "right"}:
            return {"winner": paper_b, "loser": paper_a}
        paper_a_id = _paper_id(paper_a, "paper_a")
        paper_b_id = _paper_id(paper_b, "paper_b")
        if winner_key == paper_a_id.lower():
            return {"winner": paper_a, "loser": paper_b}
        if winner_key == paper_b_id.lower():
            return {"winner": paper_b, "loser": paper_a}
    raise ValueError(f"SciImpact record on line {line_number} is not a supported pair")


def _upsert_paper(
    papers: OrderedDict[str, dict[str, Any]],
    source: dict[str, Any],
    paper_id: str,
    *,
    pair_id: str,
    role: str,
    impact_dimension: str | None,
    field: str | None,
) -> None:
    if paper_id not in papers:
        papers[paper_id] = {
            "paper_id": paper_id,
            "title": _title(source, paper_id),
            "abstract": _optional_string(source.get("abstract")),
            "year": _optional_int(source.get("year")),
            "venue": _optional_string(source.get("venue")),
            "field": _optional_string(source.get("field")) or field,
            "references": [],
            "metadata": {
                "source_benchmark": "sciimpact",
                "sciimpact_pair_ids": [],
                "sciimpact_roles": [],
                "sciimpact_impact_dimensions": [],
                "sciimpact_fields": [],
                "external_ids": _external_ids(source),
            },
        }
    paper = papers[paper_id]
    metadata = paper["metadata"]
    _append_once(metadata["sciimpact_pair_ids"], pair_id)
    _append_once(metadata["sciimpact_roles"], role)
    if impact_dimension is not None:
        _append_once(metadata["sciimpact_impact_dimensions"], impact_dimension)
    if field is not None:
        _append_once(metadata["sciimpact_fields"], field)


def _paper_id(source: dict[str, Any], fallback: str) -> str:
    for key in ("paper_id", "id", "corpus_id", "s2_id", "semantic_scholar_id"):
        value = _optional_string(source.get(key))
        if value is not None:
            return value
    doi = _optional_string(source.get("doi") or source.get("DOI"))
    if doi is not None:
        return f"doi:{doi.lower()}"
    return fallback


def _title(source: dict[str, Any], paper_id: str) -> str:
    value = _optional_string(source.get("title"))
    return value if value is not None else f"SciImpact paper {paper_id}"


def _first_string(record: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = _optional_string(record.get(key))
        if value is not None:
            return value
    return None


def _external_ids(source: dict[str, Any]) -> dict[str, str]:
    external: dict[str, str] = {}
    for source_key, output_key in [
        ("doi", "doi"),
        ("DOI", "doi"),
        ("corpus_id", "corpus_id"),
        ("s2_id", "s2_id"),
        ("semantic_scholar_id", "s2_id"),
    ]:
        value = _optional_string(source.get(source_key))
        if value is not None:
            external[output_key] = value
    return external


def _append_once(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped if stripped else None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
