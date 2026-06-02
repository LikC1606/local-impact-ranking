from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from local_impact_ranking.io import load_papers_jsonl


EXTERNAL_IMPACT_ALIASES = {
    "field_year_percentile": (
        "field_year_percentile",
        "openalex_field_year_percentile",
        "field_year_citation_percentile",
        "citation_normalized_percentile",
        "citation_percentile",
        "openalex_percentile",
    ),
    "openalex_fwci": (
        "openalex_fwci",
        "fwci",
        "field_weighted_citation_impact",
    ),
    "bip_impact_percentile": (
        "bip_impact_percentile",
        "bip_percentile",
        "bip_field_year_percentile",
    ),
    "bip_impact_score": (
        "bip_impact_score",
        "bip_score",
        "bip_influence_score",
    ),
    "bip_impact_class": (
        "bip_impact_class",
        "bip_class",
    ),
    "bip_pagerank": (
        "bip_pagerank",
        "bip_page_rank",
    ),
    "bip_att_rank": (
        "bip_att_rank",
        "bip_attrank",
    ),
    "bip_ram": (
        "bip_ram",
        "bip_ram_score",
    ),
    "semantic_scholar_influential_citation_count": (
        "semantic_scholar_influential_citation_count",
        "s2_influential_citation_count",
        "influential_citation_count",
        "highly_influential_citation_count",
    ),
    "relative_citation_ratio": (
        "relative_citation_ratio",
        "nih_rcr",
        "rcr",
        "icite_rcr",
        "relative_citation_ratio_value",
    ),
    "pubmed_citation_count": (
        "pubmed_citation_count",
        "pmid_citation_count",
        "pmc_citation_count",
    ),
    "patent_citations": (
        "patent_citations",
        "patent_citation_count",
        "patent_count",
    ),
    "policy_citations": (
        "policy_citations",
        "policy_citation_count",
        "policy_mentions",
        "guideline_citations",
        "clinical_guideline_citations",
        "practice_guideline_citations",
        "clinical_guideline_mentions",
        "guideline_mentions",
        "consensus_guideline_citations",
    ),
    "life_science_translation": (
        "life_science_translation",
        "life_science_translation_count",
        "biomedical_translation_count",
    ),
    "clinical_trial_citations": (
        "clinical_trial_citations",
        "clinical_trial_mentions",
        "clinical_study_citations",
        "trial_registry_citations",
    ),
    "drug_approval_mentions": (
        "drug_approval_mentions",
        "fda_label_mentions",
        "ema_label_mentions",
    ),
    "media_mentions": (
        "media_mentions",
        "news_mentions",
        "press_mentions",
        "altmetric_mentions",
    ),
    "award_count": (
        "award_count",
        "awards",
        "best_paper_awards",
        "prize_count",
    ),
    "artifact_adoption": (
        "artifact_adoption",
        "artifact_adoption_count",
        "software_mentions",
        "software_citations",
        "github_stars",
        "repository_stars",
        "dataset_reuses",
        "dataset_citations",
        "model_downloads",
        "dataset_downloads",
        "huggingface_downloads",
    ),
    "protocol_reuses": (
        "protocol_reuses",
        "protocol_citations",
    ),
    "assay_reuses": (
        "assay_reuses",
        "assay_citations",
    ),
    "reagent_reuses": (
        "reagent_reuses",
        "reagent_citations",
    ),
    "reagent_or_assay_reuses": (
        "reagent_or_assay_reuses",
    ),
    "antibody_reuses": (
        "antibody_reuses",
    ),
    "plasmid_reuses": (
        "plasmid_reuses",
    ),
    "cell_line_reuses": (
        "cell_line_reuses",
    ),
    "mouse_model_reuses": (
        "mouse_model_reuses",
    ),
    "biobank_reuses": (
        "biobank_reuses",
    ),
}


def merge_external_impact_evidence(
    input_path: str | Path,
    evidence_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    papers = [paper.model_dump(mode="json", exclude_none=True) for paper in load_papers_jsonl(input_path)]
    evidence_rows = _load_evidence_rows(evidence_path)
    evidence_by_key: dict[str, dict[str, Any]] = {}
    for row in evidence_rows:
        for key in _evidence_match_keys(row):
            evidence_by_key.setdefault(key, row)

    matched_count = 0
    unmatched_count = 0
    merged_rows: list[dict[str, Any]] = []
    for paper in papers:
        evidence = _find_evidence_for_paper(paper, evidence_by_key)
        if evidence is None:
            unmatched_count += 1
            merged_rows.append(paper)
            continue
        matched_count += 1
        merged_rows.append(_merge_evidence_row(paper, evidence))

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "paper_count": len(papers),
        "evidence_row_count": len(evidence_rows),
        "matched_count": matched_count,
        "unmatched_count": unmatched_count,
        "output_path": str(destination),
    }


def _load_evidence_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix.lower() == ".csv":
        with source.open("r", newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    rows = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid evidence JSON on line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Evidence row {line_number} must be an object")
            rows.append(payload)
    return rows


def _find_evidence_for_paper(
    paper: dict[str, Any],
    evidence_by_key: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    for key in _paper_match_keys(paper):
        evidence = evidence_by_key.get(key)
        if evidence is not None:
            return evidence
    return None


def _paper_match_keys(paper: dict[str, Any]) -> list[str]:
    metadata = paper.get("metadata") if isinstance(paper.get("metadata"), dict) else {}
    external_ids = metadata.get("external_ids") if isinstance(metadata.get("external_ids"), dict) else {}
    candidates = {
        "paper_id": paper.get("paper_id"),
        "doi": paper.get("doi") or external_ids.get("doi"),
        "openalex": paper.get("openalex") or external_ids.get("openalex"),
        "semantic_scholar": paper.get("semantic_scholar")
        or paper.get("semantic_scholar_id")
        or paper.get("s2_id")
        or external_ids.get("semantic_scholar")
        or external_ids.get("semantic_scholar_id")
        or external_ids.get("s2_id"),
    }
    return _match_keys_from_candidates(candidates)


def _evidence_match_keys(row: dict[str, Any]) -> list[str]:
    external_ids = row.get("external_ids") if isinstance(row.get("external_ids"), dict) else {}
    candidates = {
        "paper_id": row.get("paper_id") or row.get("id"),
        "doi": row.get("doi") or external_ids.get("doi"),
        "openalex": row.get("openalex")
        or row.get("openalex_id")
        or external_ids.get("openalex"),
        "semantic_scholar": row.get("semantic_scholar")
        or row.get("semantic_scholar_id")
        or row.get("s2_id")
        or row.get("corpus_id")
        or external_ids.get("semantic_scholar")
        or external_ids.get("s2_id"),
    }
    return _match_keys_from_candidates(candidates)


def _match_keys_from_candidates(candidates: dict[str, Any]) -> list[str]:
    keys = []
    for kind, value in candidates.items():
        normalized = _normalize_identifier(kind, value)
        if normalized:
            keys.append(f"{kind}:{normalized}")
    return keys


def _normalize_identifier(kind: str, value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if kind == "doi":
        text = text.lower()
        text = text.removeprefix("https://doi.org/")
        text = text.removeprefix("http://doi.org/")
        text = text.removeprefix("doi:")
    if kind == "openalex":
        text = text.rstrip("/").split("/")[-1]
    return text


def _merge_evidence_row(paper: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    record = dict(paper)
    metadata = dict(record.get("metadata") or {})
    external_ids = dict(metadata.get("external_ids") or {})
    for key in ("doi", "openalex", "openalex_id", "semantic_scholar_id", "s2_id"):
        value = evidence.get(key)
        if value in (None, ""):
            continue
        if key == "openalex_id":
            external_ids["openalex"] = str(value)
        elif key in {"semantic_scholar_id", "s2_id"}:
            external_ids["semantic_scholar"] = str(value)
        else:
            external_ids[key] = str(value)
    if external_ids:
        metadata["external_ids"] = external_ids

    citation_count = _numeric(
        evidence.get("citation_count")
        or evidence.get("cited_by_count")
        or evidence.get("pubmed_citation_count")
        or evidence.get("pmc_citation_count")
    )
    if citation_count is not None:
        record["citation_count"] = citation_count
        metadata["citation_count"] = citation_count

    external_impact = dict(metadata.get("external_impact") or {})
    nested_impact = evidence.get("external_impact")
    if isinstance(nested_impact, dict):
        external_impact.update(nested_impact)
    for output_key, aliases in EXTERNAL_IMPACT_ALIASES.items():
        value = _first_numeric(evidence, aliases)
        if value is not None:
            external_impact[output_key] = value
    if external_impact:
        metadata["external_impact"] = external_impact

    sources = set(metadata.get("external_evidence_sources") or [])
    if any(key in external_impact for key in ("field_year_percentile", "openalex_fwci")):
        metadata["openalex_enriched"] = True
        sources.add("openalex")
    if any(key.startswith("bip_") for key in external_impact):
        metadata["bip_enriched"] = True
        sources.add("bip")
    if "semantic_scholar_influential_citation_count" in external_impact:
        metadata["semantic_scholar_enriched"] = True
        sources.add("semantic_scholar")
    if "relative_citation_ratio" in external_impact:
        metadata["icite_enriched"] = True
        sources.add("icite")
    if "pubmed_citation_count" in external_impact:
        metadata["pubmed_enriched"] = True
        sources.add("pubmed")
    if any(
        key in external_impact
        for key in (
            "life_science_translation",
            "clinical_trial_citations",
            "drug_approval_mentions",
        )
    ):
        metadata["life_science_translation_enriched"] = True
        sources.add("life_science_translation")
    if sources:
        metadata["external_evidence_sources"] = sorted(sources)

    record["metadata"] = metadata
    return record


def _first_numeric(row: dict[str, Any], aliases: tuple[str, ...]) -> float | None:
    for key in aliases:
        value = _numeric(row.get(key))
        if value is not None:
            return value
    return None


def _numeric(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(number, 0.0)
