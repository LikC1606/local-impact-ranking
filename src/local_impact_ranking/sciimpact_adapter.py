from __future__ import annotations

import json
import re
import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any

_SFT_ITEM_MARKERS = (
    ("paper", "Paper A:", "Paper B:"),
    ("dataset", "Dataset A:", "Dataset B:"),
    ("github_repo", "GitHub repo A README:", "GitHub repo B README:"),
    ("model", "Model A:", "Model B:"),
)


def convert_sciimpact_jsonl(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    skip_invalid: bool = False,
) -> dict[str, Any]:
    source = Path(input_path)
    destination = Path(output_dir)
    papers: OrderedDict[str, dict[str, Any]] = OrderedDict()
    labels: list[dict[str, Any]] = []
    skipped_pairs: list[dict[str, Any]] = []
    source_record_count = 0

    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            source_record_count += 1
            record = json.loads(stripped)
            try:
                parsed = _parse_pair(record, line_number)
            except ValueError as exc:
                if not skip_invalid:
                    raise
                skipped_pairs.append(
                    {
                        "source_line": line_number,
                        "error": str(exc),
                    }
                )
                continue
            winner = parsed["winner"]
            loser = parsed["loser"]
            winner_id = _paper_id(winner, f"line-{line_number}-winner")
            loser_id = _paper_id(loser, f"line-{line_number}-loser")
            pair_id = str(record.get("id") or record.get("pair_id") or f"sciimpact-{line_number}")
            impact_dimension = _first_string(
                record,
                ("impact_dimension", "dimension", "impact_type", "category", "task"),
            ) or parsed.get("impact_dimension")
            field = _first_string(record, ("field", "domain", "discipline", "subject"))
            item_type = parsed.get("item_type")

            _upsert_paper(
                papers,
                winner,
                winner_id,
                pair_id=pair_id,
                role="winner",
                impact_dimension=impact_dimension,
                field=field,
                item_type=item_type,
            )
            _upsert_paper(
                papers,
                loser,
                loser_id,
                pair_id=pair_id,
                role="loser",
                impact_dimension=impact_dimension,
                field=field,
                item_type=item_type,
            )
            labels.append(
                {
                    "pair_id": pair_id,
                    "winner_id": winner_id,
                    "loser_id": loser_id,
                    "impact_dimension": impact_dimension or "unknown",
                    "field": field or "unknown",
                    "item_type": item_type or "paper",
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
        "source_record_count": source_record_count,
        "paper_count": len(papers),
        "pair_count": len(labels),
        "skipped_pair_count": len(skipped_pairs),
        "skipped_pairs": skipped_pairs[:20],
        "papers_path": str(papers_path),
        "pairwise_labels_path": str(labels_path),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parse_pair(record: dict[str, Any], line_number: int) -> dict[str, dict[str, Any]]:
    if isinstance(record.get("conversations"), list):
        return _parse_sharegpt_pair(record, line_number)
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


def _parse_sharegpt_pair(record: dict[str, Any], line_number: int) -> dict[str, Any]:
    conversations = record["conversations"]
    system = _conversation_value(conversations, "system")
    human = _conversation_value(conversations, "human")
    answer = _conversation_value(conversations, "gpt")
    if human is None or answer is None:
        raise ValueError(f"SciImpact SFT record on line {line_number} is missing human/gpt text")

    item_type, paper_a, paper_b = _parse_sft_items(human, line_number)
    winner_key = _parse_sft_winner(answer, line_number)
    impact_dimension = _infer_sft_impact_dimension(system or human)

    paper_a["paper_id"] = _sft_paper_id(paper_a, item_type)
    paper_b["paper_id"] = _sft_paper_id(paper_b, item_type)
    _sft_metadata(paper_a, item_type, line_number, "a")
    _sft_metadata(paper_b, item_type, line_number, "b")

    if winner_key == "a":
        winner, loser = paper_a, paper_b
    else:
        winner, loser = paper_b, paper_a
    return {
        "winner": winner,
        "loser": loser,
        "impact_dimension": impact_dimension,
        "item_type": item_type,
    }


def _conversation_value(conversations: list[Any], role: str) -> str | None:
    for message in conversations:
        if not isinstance(message, dict):
            continue
        if message.get("from") == role:
            return _optional_string(message.get("value"))
    return None


def _parse_sft_items(human_prompt: str, line_number: int) -> tuple[str, dict[str, Any], dict[str, Any]]:
    for item_type, marker_a, marker_b in _SFT_ITEM_MARKERS:
        if marker_a not in human_prompt or marker_b not in human_prompt:
            continue
        before_b, raw_b = human_prompt.split(marker_b, 1)
        _, raw_a = before_b.split(marker_a, 1)
        return (
            item_type,
            _sft_item_record(raw_a, item_type),
            _sft_item_record(raw_b, item_type),
        )
    raise ValueError(f"SciImpact SFT record on line {line_number} has no supported A/B markers")


def _sft_item_record(raw_text: str, item_type: str) -> dict[str, Any]:
    text = _strip_sft_question_suffix(raw_text)
    if item_type == "paper":
        title = _extract_between(text, "Title:", ("Year:", "Abstract:"))
        abstract = _extract_after(text, "Abstract:")
        year_text = _extract_between(text, "Year:", ("Abstract:",))
        return {
            "title": title or "Untitled SciImpact paper",
            "abstract": abstract,
            "year": _optional_int(year_text),
            "metadata": {"sciimpact_raw_text": text},
        }
    if item_type in {"dataset", "model"}:
        name = _extract_between(text, "Name:", ("\n",))
        return {
            "title": name or _first_nonempty_line(text) or f"Untitled SciImpact {item_type}",
            "abstract": text,
            "metadata": {"sciimpact_raw_text": text},
        }
    return {
        "title": _first_nonempty_line(text) or "Untitled SciImpact GitHub repo",
        "abstract": text,
        "metadata": {"sciimpact_raw_text": text},
    }


def _parse_sft_winner(answer: str, line_number: int) -> str:
    match = re.search(r"\b(?:Paper|Dataset|Model|GitHub repo)\s+([AB])\b", answer, re.IGNORECASE)
    if match is None:
        raise ValueError(f"SciImpact SFT record on line {line_number} has no A/B answer")
    return match.group(1).lower()


def _infer_sft_impact_dimension(text: str) -> str:
    lowered = text.lower()
    if "citation" in lowered:
        return "citation"
    if "media mention" in lowered:
        return "media"
    if "patent" in lowered:
        return "patent"
    if "download" in lowered:
        return "downloads"
    if "star" in lowered:
        return "stars"
    if "best paper" in lowered or "nobel" in lowered or "award" in lowered:
        return "award"
    return "unknown"


def _strip_sft_question_suffix(text: str) -> str:
    suffix_patterns = (
        r"\s+Which paper\b.*$",
        r"\s+Which dataset\b.*$",
        r"\s+Which repository\b.*$",
        r"\s+Which model\b.*$",
        r"\s+Reply with exactly one sentence\b.*$",
    )
    stripped = text.strip()
    for pattern in suffix_patterns:
        stripped = re.sub(pattern, "", stripped, flags=re.IGNORECASE | re.DOTALL).strip()
    return stripped


def _extract_between(text: str, start_marker: str, end_markers: tuple[str, ...]) -> str | None:
    start = text.find(start_marker)
    if start < 0:
        return None
    value_start = start + len(start_marker)
    value_end = len(text)
    for marker in end_markers:
        marker_index = text.find(marker, value_start)
        if marker_index >= 0:
            value_end = min(value_end, marker_index)
    return _optional_string(text[value_start:value_end])


def _extract_after(text: str, marker: str) -> str | None:
    marker_index = text.find(marker)
    if marker_index < 0:
        return None
    return _optional_string(text[marker_index + len(marker) :])


def _first_nonempty_line(text: str) -> str | None:
    for line in text.splitlines():
        value = _optional_string(line)
        if value is not None:
            return value
    return _optional_string(text)


def _sft_metadata(
    item: dict[str, Any],
    item_type: str,
    line_number: int,
    side: str,
) -> None:
    metadata = item.setdefault("metadata", {})
    metadata["sciimpact_item_type"] = item_type
    metadata["sciimpact_source_line"] = line_number
    metadata["sciimpact_source_side"] = side


def _sft_paper_id(item: dict[str, Any], item_type: str) -> str:
    title = _optional_string(item.get("title")) or ""
    abstract = _optional_string(item.get("abstract")) or ""
    year = _optional_int(item.get("year"))
    key = json.dumps(
        {
            "item_type": item_type,
            "title": title.casefold(),
            "abstract": abstract.casefold(),
            "year": year,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"sciimpact:{item_type}:{digest}"


def _upsert_paper(
    papers: OrderedDict[str, dict[str, Any]],
    source: dict[str, Any],
    paper_id: str,
    *,
    pair_id: str,
    role: str,
    impact_dimension: str | None,
    field: str | None,
    item_type: str | None,
) -> None:
    if paper_id not in papers:
        source_metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
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
                "sciimpact_item_types": [],
                "external_ids": _external_ids(source),
                **source_metadata,
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
    if item_type is not None:
        _append_once(metadata["sciimpact_item_types"], item_type)


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
