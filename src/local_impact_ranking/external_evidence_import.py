from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from local_impact_ranking.csv_safety import sanitize_csv_row
from local_impact_ranking.io import load_papers_jsonl
from local_impact_ranking.normalize import normalize_doi


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
    "ucite_citation_count": (
        "ucite_citation_count",
        "ucite_cited_by_count",
        "ucite_citations",
        "ucite_total_citations",
    ),
    "nih_occ_citation_count": (
        "nih_occ_citation_count",
        "nih_occ_cited_by_count",
        "nih_occ_citations",
        "nih_open_citation_collection_citation_count",
        "open_citation_collection_citation_count",
        "occ_citation_count",
    ),
    "dimensions_citation_count": (
        "dimensions_citation_count",
        "dimensions_cited_by_count",
        "dimensions_citations",
        "dimensions_times_cited",
    ),
    "scopus_citation_count": (
        "scopus_citation_count",
        "scopus_cited_by_count",
        "scopus_citations",
        "scopus_times_cited",
    ),
    "web_of_science_citation_count": (
        "web_of_science_citation_count",
        "web_of_science_cited_by_count",
        "web_of_science_citations",
        "webofscience_citation_count",
        "wos_citation_count",
        "wos_times_cited",
        "times_cited",
    ),
    "lens_citation_count": (
        "lens_citation_count",
        "lens_cited_by_count",
        "lens_citations",
    ),
    "google_scholar_citation_count": (
        "google_scholar_citation_count",
        "google_scholar_cited_by_count",
        "google_scholar_citations",
        "scholar_citation_count",
        "gs_citation_count",
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


EXTERNAL_METADATA_ALIASES = {
    "scopus_title": ("scopus_title",),
    "web_of_science_title": (
        "web_of_science_title",
        "webofscience_title",
        "wos_title",
    ),
    "dimensions_title": ("dimensions_title",),
    "lens_title": ("lens_title",),
    "google_scholar_title": (
        "google_scholar_title",
        "scholar_title",
        "gs_title",
    ),
    "ucite_title": ("ucite_title",),
    "nih_occ_title": (
        "nih_occ_title",
        "nih_open_citation_collection_title",
        "occ_title",
    ),
    "external_identity_status": ("external_identity_status",),
}


EXTERNAL_EVIDENCE_TEMPLATE_FIELDS = (
    "paper_id",
    "title",
    "doi",
    "openalex",
    "semantic_scholar_id",
    "pmid",
    "pmcid",
    "openalex_cited_by_count",
    "semantic_scholar_citation_count",
    "pubmed_citation_count",
    "icite_citation_count",
    "ucite_citation_count",
    "nih_occ_citation_count",
    "dimensions_citation_count",
    "scopus_citation_count",
    "web_of_science_citation_count",
    "lens_citation_count",
    "google_scholar_citation_count",
    "opencitations_cited_by_count",
    "crossref_is_referenced_by_count",
    "crossref_data_citation_count",
    "datacite_citation_count",
    "datacite_events_cited_by_count",
    "openaire_citation_count",
    "wikidata_cited_by_count",
    "europepmc_cited_by_count",
    "ncbi_pubmed_citedin_count",
    "scopus_title",
    "web_of_science_title",
    "dimensions_title",
    "lens_title",
    "google_scholar_title",
    "ucite_title",
    "nih_occ_title",
    "external_identity_status",
    "source_note",
)


EXTERNAL_EVIDENCE_TEMPLATE_PREFILL_ALIASES = {
    "openalex_cited_by_count": (
        "openalex_cited_by_count",
        "openalex_citation_count",
        "openalex_citations",
    ),
    "semantic_scholar_citation_count": (
        "semantic_scholar_citation_count",
        "s2_citation_count",
        "semantic_scholar_citations",
    ),
    "pubmed_citation_count": (
        "pubmed_citation_count",
        "pmid_citation_count",
        "pmc_citation_count",
    ),
    "icite_citation_count": (
        "icite_citation_count",
        "nih_icite_citation_count",
        "icite_total_citations",
    ),
    "ucite_citation_count": EXTERNAL_IMPACT_ALIASES["ucite_citation_count"],
    "nih_occ_citation_count": EXTERNAL_IMPACT_ALIASES["nih_occ_citation_count"],
    "dimensions_citation_count": EXTERNAL_IMPACT_ALIASES[
        "dimensions_citation_count"
    ],
    "scopus_citation_count": EXTERNAL_IMPACT_ALIASES["scopus_citation_count"],
    "web_of_science_citation_count": EXTERNAL_IMPACT_ALIASES[
        "web_of_science_citation_count"
    ],
    "lens_citation_count": EXTERNAL_IMPACT_ALIASES["lens_citation_count"],
    "google_scholar_citation_count": EXTERNAL_IMPACT_ALIASES[
        "google_scholar_citation_count"
    ],
    "opencitations_cited_by_count": (
        "opencitations_cited_by_count",
        "opencitations_citation_count",
        "opencitations_citations",
    ),
    "crossref_is_referenced_by_count": (
        "crossref_is_referenced_by_count",
        "crossref_cited_by_count",
        "is_referenced_by_count",
        "is-referenced-by-count",
    ),
    "crossref_data_citation_count": (
        "crossref_data_citation_count",
        "crossref_data_citations_total_results",
    ),
    "datacite_citation_count": (
        "datacite_citation_count",
        "datacite_cited_by_count",
    ),
    "datacite_events_cited_by_count": (
        "datacite_events_cited_by_count",
        "datacite_events_citation_count",
    ),
    "openaire_citation_count": (
        "openaire_citation_count",
        "openaire_cited_by_count",
    ),
    "wikidata_cited_by_count": (
        "wikidata_cited_by_count",
        "wikidata_citation_count",
    ),
    "europepmc_cited_by_count": (
        "europepmc_cited_by_count",
        "europepmc_citation_count",
    ),
    "ncbi_pubmed_citedin_count": (
        "ncbi_pubmed_citedin_count",
        "ncbi_pubmed_cited_by_count",
    ),
}


def merge_external_impact_evidence(
    input_path: str | Path,
    evidence_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    papers = [paper.model_dump(mode="json", exclude_none=True) for paper in load_papers_jsonl(input_path)]
    evidence_rows = _load_evidence_rows(evidence_path)
    evidence_validation = _evidence_validation_report(evidence_rows)
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
        "evidence_validation": evidence_validation,
        "output_path": str(destination),
    }


def write_external_evidence_template(
    input_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    papers = [paper.model_dump(mode="json", exclude_none=True) for paper in load_papers_jsonl(input_path)]
    rows = [_external_evidence_template_row(paper) for paper in papers]

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EXTERNAL_EVIDENCE_TEMPLATE_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(sanitize_csv_row(row))

    papers_with_any_query_id = sum(1 for row in rows if _template_row_has_query_id(row))
    return {
        "paper_count": len(rows),
        "rows_written": len(rows),
        "papers_with_any_query_id": papers_with_any_query_id,
        "papers_without_query_id": len(rows) - papers_with_any_query_id,
        "template_field_count": len(EXTERNAL_EVIDENCE_TEMPLATE_FIELDS),
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


def _external_evidence_template_row(paper: dict[str, Any]) -> dict[str, Any]:
    metadata = paper.get("metadata") if isinstance(paper.get("metadata"), dict) else {}
    external_ids = metadata.get("external_ids") if isinstance(metadata.get("external_ids"), dict) else {}
    row = {field: "" for field in EXTERNAL_EVIDENCE_TEMPLATE_FIELDS}
    row.update(
        {
            "paper_id": _text(paper.get("paper_id")),
            "title": _text(paper.get("title")),
            "doi": _template_doi(paper.get("doi") or external_ids.get("doi")),
            "openalex": _text(
                paper.get("openalex")
                or paper.get("openalex_id")
                or external_ids.get("openalex")
                or external_ids.get("openalex_id")
            ),
            "semantic_scholar_id": _text(
                paper.get("semantic_scholar")
                or paper.get("semantic_scholar_id")
                or paper.get("s2_id")
                or external_ids.get("semantic_scholar")
                or external_ids.get("semantic_scholar_id")
                or external_ids.get("s2_id")
            ),
            "pmid": _text(
                paper.get("pmid")
                or paper.get("pubmed_id")
                or metadata.get("pmid")
                or metadata.get("pubmed_id")
                or external_ids.get("pmid")
                or external_ids.get("pubmed_id")
            ),
            "pmcid": _text(
                paper.get("pmcid")
                or paper.get("pmc_id")
                or metadata.get("pmcid")
                or metadata.get("pmc_id")
                or external_ids.get("pmcid")
                or external_ids.get("pmc_id")
            ),
        }
    )
    _prefill_template_evidence(row, paper, metadata)
    return row


def _evidence_validation_report(evidence_rows: list[dict[str, Any]]) -> dict[str, Any]:
    match_key_counts: Counter[str] = Counter()
    rows_without_match_key: list[dict[str, Any]] = []
    invalid_numeric_cells: list[dict[str, Any]] = []
    negative_numeric_cells: list[dict[str, Any]] = []
    numeric_fields = _numeric_evidence_fields()

    for row_index, row in enumerate(evidence_rows, start=1):
        match_keys = _evidence_match_keys(row)
        if not match_keys:
            _append_issue_sample(
                rows_without_match_key,
                {"row_number": row_index},
            )
        for match_key in match_keys:
            match_key_counts[match_key] += 1
        for field, value in row.items():
            if field not in numeric_fields or value in (None, ""):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                _append_issue_sample(
                    invalid_numeric_cells,
                    {
                        "row_number": row_index,
                        "field": field,
                        "value": str(value),
                    },
                )
                continue
            if number < 0.0:
                _append_issue_sample(
                    negative_numeric_cells,
                    {
                        "row_number": row_index,
                        "field": field,
                        "value": str(value),
                    },
                )

    duplicate_match_keys = [
        {"match_key": key, "count": count}
        for key, count in sorted(match_key_counts.items())
        if count > 1
    ]
    duplicate_match_key_count = len(duplicate_match_keys)
    rows_without_match_key_count = sum(
        1 for row in evidence_rows if not _evidence_match_keys(row)
    )
    invalid_numeric_cell_count = _numeric_issue_count(
        evidence_rows,
        numeric_fields,
        issue="invalid",
    )
    negative_numeric_cell_count = _numeric_issue_count(
        evidence_rows,
        numeric_fields,
        issue="negative",
    )
    issue_count = (
        duplicate_match_key_count
        + rows_without_match_key_count
        + invalid_numeric_cell_count
        + negative_numeric_cell_count
    )
    return {
        "evidence_row_count": len(evidence_rows),
        "issue_count": issue_count,
        "has_issues": issue_count > 0,
        "rows_without_match_key_count": rows_without_match_key_count,
        "duplicate_match_key_count": duplicate_match_key_count,
        "invalid_numeric_cell_count": invalid_numeric_cell_count,
        "negative_numeric_cell_count": negative_numeric_cell_count,
        "rows_without_match_key": rows_without_match_key,
        "duplicate_match_keys": duplicate_match_keys[:50],
        "invalid_numeric_cells": invalid_numeric_cells,
        "negative_numeric_cells": negative_numeric_cells,
    }


def _numeric_evidence_fields() -> set[str]:
    fields: set[str] = {"citation_count", "cited_by_count", "citations"}
    for aliases in EXTERNAL_IMPACT_ALIASES.values():
        fields.update(aliases)
    for aliases in EXTERNAL_EVIDENCE_TEMPLATE_PREFILL_ALIASES.values():
        fields.update(aliases)
    return fields


def _append_issue_sample(samples: list[dict[str, Any]], issue: dict[str, Any]) -> None:
    if len(samples) < 50:
        samples.append(issue)


def _numeric_issue_count(
    evidence_rows: list[dict[str, Any]],
    numeric_fields: set[str],
    *,
    issue: str,
) -> int:
    count = 0
    for row in evidence_rows:
        for field, value in row.items():
            if field not in numeric_fields or value in (None, ""):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                if issue == "invalid":
                    count += 1
                continue
            if issue == "negative" and number < 0.0:
                count += 1
    return count


def _template_doi(value: Any) -> str:
    if value in (None, ""):
        return ""
    return normalize_doi(value)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _template_row_has_query_id(row: dict[str, Any]) -> bool:
    return any(row.get(key) for key in ("doi", "openalex", "semantic_scholar_id", "pmid", "pmcid"))


def _prefill_template_evidence(
    row: dict[str, str],
    paper: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    sources = _metric_sources_for_template(paper, metadata)
    for output_key, aliases in EXTERNAL_EVIDENCE_TEMPLATE_PREFILL_ALIASES.items():
        value = _first_numeric_from_sources(sources, aliases)
        if value is not None:
            row[output_key] = _format_template_number(value)
    for output_key, aliases in EXTERNAL_METADATA_ALIASES.items():
        if output_key not in row:
            continue
        value = _first_text_from_sources(sources, aliases)
        if value is not None:
            row[output_key] = value


def _metric_sources_for_template(
    paper: dict[str, Any],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        source
        for source in (
            metadata.get("external_impact"),
            metadata.get("impact"),
            metadata.get("metrics"),
            metadata,
            paper,
        )
        if isinstance(source, dict)
    ]


def _first_numeric_from_sources(
    sources: list[dict[str, Any]],
    aliases: tuple[str, ...],
) -> float | None:
    for source in sources:
        value = _first_numeric(source, aliases)
        if value is not None:
            return value
    return None


def _first_text_from_sources(
    sources: list[dict[str, Any]],
    aliases: tuple[str, ...],
) -> str | None:
    for source in sources:
        value = _first_text(source, aliases)
        if value is not None:
            return value
    return None


def _format_template_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value)


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
        "pmid": paper.get("pmid")
        or paper.get("pubmed_id")
        or external_ids.get("pmid")
        or external_ids.get("pubmed_id"),
        "pmcid": paper.get("pmcid")
        or paper.get("pmc_id")
        or external_ids.get("pmcid")
        or external_ids.get("pmc_id"),
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
        "pmid": row.get("pmid")
        or row.get("pubmed_id")
        or external_ids.get("pmid")
        or external_ids.get("pubmed_id"),
        "pmcid": row.get("pmcid")
        or row.get("pmc_id")
        or external_ids.get("pmcid")
        or external_ids.get("pmc_id"),
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
        text = normalize_doi(text).lower()
    if kind == "openalex":
        text = text.rstrip("/").split("/")[-1]
    if kind == "pmid":
        text = text.lower().removeprefix("pmid:").strip()
    if kind == "pmcid":
        text = text.lower().removeprefix("pmcid:").strip().upper()
    return text


def _merge_evidence_row(paper: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    record = dict(paper)
    metadata = dict(record.get("metadata") or {})
    external_ids = dict(metadata.get("external_ids") or {})
    for key in (
        "doi",
        "openalex",
        "openalex_id",
        "semantic_scholar_id",
        "s2_id",
        "pmid",
        "pubmed_id",
        "pmcid",
        "pmc_id",
    ):
        value = evidence.get(key)
        if value in (None, ""):
            continue
        if key == "openalex_id":
            external_ids["openalex"] = str(value)
        elif key in {"semantic_scholar_id", "s2_id"}:
            external_ids["semantic_scholar"] = str(value)
        elif key == "pubmed_id":
            external_ids["pmid"] = str(value)
        elif key == "pmc_id":
            external_ids["pmcid"] = str(value)
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

    for output_key, aliases in EXTERNAL_METADATA_ALIASES.items():
        value = _first_text(evidence, aliases)
        if value is not None:
            metadata[output_key] = value

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
    high_recall_sources = {
        "ucite_citation_count": ("ucite", "ucite_enriched"),
        "nih_occ_citation_count": ("nih_occ", "nih_occ_enriched"),
        "dimensions_citation_count": ("dimensions", "dimensions_enriched"),
        "scopus_citation_count": ("scopus", "scopus_enriched"),
        "web_of_science_citation_count": (
            "web_of_science",
            "web_of_science_enriched",
        ),
        "lens_citation_count": ("lens", "lens_enriched"),
        "google_scholar_citation_count": (
            "google_scholar",
            "google_scholar_enriched",
        ),
    }
    for impact_key, (source_name, enriched_key) in high_recall_sources.items():
        if impact_key in external_impact:
            metadata[enriched_key] = True
            metadata["high_recall_citation_enriched"] = True
            sources.add(source_name)
            sources.add("high_recall_citation")
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


def _first_text(row: dict[str, Any], aliases: tuple[str, ...]) -> str | None:
    for key in aliases:
        value = row.get(key)
        if value in (None, ""):
            continue
        return str(value).strip()
    return None


def _numeric(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(number, 0.0)
