from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from local_impact_ranking.models import Author, Paper, Reference


def entity_resolution_report(papers: list[Paper]) -> dict[str, Any]:
    references = [reference for paper in papers for reference in paper.references]
    author_mentions = []
    reference_author_mentions = []
    for paper in papers:
        author_mentions.extend(("target", paper.paper_id, author) for author in paper.authors)
        for reference in paper.references:
            reference_mentions = [
                ("reference", reference.reference_id or "", author)
                for author in reference.authors
            ]
            author_mentions.extend(reference_mentions)
            reference_author_mentions.extend(reference_mentions)

    same_name_multiple_ids = _same_name_multiple_ids(author_mentions)
    same_id_multiple_names = _same_id_multiple_names(author_mentions)
    institution_variants = _institution_variants(author_mentions)
    duplicate_titles = _duplicate_reference_titles(references)
    author_id_coverage = _coverage(
        reference_author_mentions,
        lambda mention: bool(_author_id(mention[2])),
    )
    affiliation_coverage = _coverage(
        reference_author_mentions,
        lambda mention: bool(mention[2].affiliations),
    )
    quality_gate = _quality_gate(
        author_id_coverage=author_id_coverage,
        affiliation_coverage=affiliation_coverage,
        same_name_multiple_id_count=len(same_name_multiple_ids),
        same_id_multiple_name_count=len(same_id_multiple_names),
    )

    return {
        "summary": {
            "paper_count": len(papers),
            "reference_count": len(references),
            "author_mention_count": len(author_mentions),
            "reference_author_mention_count": len(reference_author_mentions),
            "author_id_coverage": author_id_coverage,
            "affiliation_coverage": affiliation_coverage,
            "same_name_multiple_id_count": len(same_name_multiple_ids),
            "same_id_multiple_name_count": len(same_id_multiple_names),
            "institution_name_variant_count": len(institution_variants),
            "duplicate_reference_title_count": len(duplicate_titles),
        },
        "same_name_multiple_ids": same_name_multiple_ids,
        "same_id_multiple_names": same_id_multiple_names,
        "institution_name_variants": institution_variants,
        "duplicate_reference_titles": duplicate_titles,
        "quality_gate": quality_gate,
        "limitations": [
            "This report flags entity-resolution risks; it does not perform external-authority disambiguation.",
            "Author and institution rankings should remain exploratory when IDs or affiliations are sparse.",
        ],
    }


def write_entity_resolution_report(
    papers: list[Paper],
    output_dir: str | Path,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report = entity_resolution_report(papers)
    (destination / "entity_resolution_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "entity_resolution_report.md").write_text(
        render_entity_resolution_markdown(report),
        encoding="utf-8",
    )
    return report


def render_entity_resolution_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    gate = report.get("quality_gate", {})
    lines = [
        "# Entity Resolution Report",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in [
        "paper_count",
        "reference_count",
        "author_mention_count",
        "reference_author_mention_count",
        "author_id_coverage",
        "affiliation_coverage",
        "same_name_multiple_id_count",
        "same_id_multiple_name_count",
        "institution_name_variant_count",
        "duplicate_reference_title_count",
    ]:
        lines.append(f"| {key} | {_format_value(summary.get(key))} |")
    lines.extend(
        [
            "",
            "## Quality Gate",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| entity_rankings_scope | {gate.get('entity_rankings_scope', '')} |",
            f"| reason | {gate.get('reason', '')} |",
            "",
            "## Same Name With Multiple IDs",
            "",
            "| Name | IDs | Mentions |",
            "| --- | --- | ---: |",
        ]
    )
    _append_name_id_rows(lines, report.get("same_name_multiple_ids", []), key="name")
    lines.extend(
        [
            "",
            "## Same ID With Multiple Names",
            "",
            "| ID | Names | Mentions |",
            "| --- | --- | ---: |",
        ]
    )
    _append_name_id_rows(lines, report.get("same_id_multiple_names", []), key="id")
    lines.extend(
        [
            "",
            "## Institution Name Variants",
            "",
            "| Normalized | Variants | Mentions |",
            "| --- | --- | ---: |",
        ]
    )
    for row in report.get("institution_name_variants", []):
        lines.append(
            f"| {row.get('normalized_name', '')} | {_join(row.get('variants', []))} | "
            f"{row.get('mention_count', 0)} |"
        )
    if not report.get("institution_name_variants"):
        lines.append("|  |  |  |")
    lines.append("")
    return "\n".join(lines)


def _same_name_multiple_ids(author_mentions: list[tuple[str, str, Author]]) -> list[dict[str, Any]]:
    ids_by_name: defaultdict[str, set[str]] = defaultdict(set)
    counts_by_name: defaultdict[str, int] = defaultdict(int)
    for _scope, _entity_id, author in author_mentions:
        name = _normalize_name(author.name)
        if not name:
            continue
        counts_by_name[name] += 1
        if author_id := _author_id(author):
            ids_by_name[name].add(author_id)
    return [
        {"name": name, "ids": sorted(ids), "mention_count": counts_by_name[name]}
        for name, ids in sorted(ids_by_name.items())
        if len(ids) > 1
    ]


def _same_id_multiple_names(author_mentions: list[tuple[str, str, Author]]) -> list[dict[str, Any]]:
    names_by_id: defaultdict[str, set[str]] = defaultdict(set)
    counts_by_id: defaultdict[str, int] = defaultdict(int)
    for _scope, _entity_id, author in author_mentions:
        author_id = _author_id(author)
        name = _normalize_name(author.name)
        if not author_id or not name:
            continue
        counts_by_id[author_id] += 1
        names_by_id[author_id].add(name)
    return [
        {"id": author_id, "names": sorted(names), "mention_count": counts_by_id[author_id]}
        for author_id, names in sorted(names_by_id.items())
        if len(names) > 1
    ]


def _institution_variants(
    author_mentions: list[tuple[str, str, Author]],
) -> list[dict[str, Any]]:
    variants_by_core: defaultdict[str, set[str]] = defaultdict(set)
    counts_by_core: defaultdict[str, int] = defaultdict(int)
    for _scope, _entity_id, author in author_mentions:
        for affiliation in author.affiliations:
            core = _institution_core(affiliation)
            if not core:
                continue
            counts_by_core[core] += 1
            variants_by_core[core].add(affiliation.strip())
    return [
        {
            "normalized_name": core,
            "variants": sorted(variants),
            "mention_count": counts_by_core[core],
        }
        for core, variants in sorted(variants_by_core.items())
        if len(variants) > 1
    ]


def _duplicate_reference_titles(references: list[Reference]) -> list[dict[str, Any]]:
    ids_by_title: defaultdict[str, set[str]] = defaultdict(set)
    for reference in references:
        title = _normalize_name(reference.title)
        if not title:
            continue
        ids_by_title[title].add(reference.reference_id or "")
    return [
        {"title": title, "reference_ids": sorted(ids), "count": len(ids)}
        for title, ids in sorted(ids_by_title.items())
        if len(ids) > 1
    ]


def _quality_gate(
    *,
    author_id_coverage: float,
    affiliation_coverage: float,
    same_name_multiple_id_count: int,
    same_id_multiple_name_count: int,
) -> dict[str, str]:
    if (
        author_id_coverage >= 0.70
        and affiliation_coverage >= 0.70
        and same_name_multiple_id_count == 0
        and same_id_multiple_name_count == 0
    ):
        return {
            "entity_rankings_scope": "reportable",
            "reason": "author IDs and affiliations have enough coverage for entity ranking.",
        }
    return {
        "entity_rankings_scope": "exploratory",
        "reason": "entity metadata has sparse coverage or ambiguity flags.",
    }


def _coverage(items: list[Any], predicate) -> float:
    if not items:
        return 0.0
    return sum(1 for item in items if predicate(item)) / len(items)


def _author_id(author: Author) -> str:
    for value in author.ids.values():
        if str(value).strip():
            return str(value).strip()
    return ""


def _normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _institution_core(value: str) -> str:
    normalized = _normalize_name(value)
    normalized = re.sub(r"\b(us|u\.s\.|united states)\b", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" ,")
    return normalized


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if value is None:
        return "0"
    return str(value)


def _append_name_id_rows(lines: list[str], rows: list[dict[str, Any]], *, key: str) -> None:
    if not rows:
        lines.append("|  |  |  |")
        return
    list_key = "ids" if key == "name" else "names"
    for row in rows[:20]:
        lines.append(
            f"| {row.get(key, '')} | {_join(row.get(list_key, []))} | "
            f"{row.get('mention_count', 0)} |"
        )


def _join(values: Any) -> str:
    return ", ".join(str(value) for value in values)
