from __future__ import annotations

import hashlib
import re
from typing import Any

from local_impact_ranking.models import Author, CitationContext, Paper, Reference

_WHITESPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi:",
)
_EXTERNAL_ID_ALIASES = {
    "doi": "doi",
    "openalex": "openalex",
    "openalexid": "openalex",
    "semanticsscholar": "semantic_scholar",
    "semanticscholar": "semantic_scholar",
    "semanticscholarid": "semantic_scholar",
    "s2": "semantic_scholar",
    "s2id": "semantic_scholar",
    "corpusid": "semantic_scholar_corpus_id",
    "pmid": "pmid",
    "pubmed": "pmid",
}
_MIN_TITLE_DEDUP_LENGTH = 20
_REFERENCE_EVIDENCE_EXTRA_FIELDS = (
    "internal_citation_match_method",
    "internal_citation_match_confidence",
    "source",
    "match_method",
    "match_confidence",
)


def normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    return _WHITESPACE.sub(" ", value).strip()


def normalize_doi(value: Any) -> str:
    normalized = normalize_text(str(value)) if value is not None else None
    if not normalized:
        return ""
    lowered = normalized.lower()
    for prefix in _DOI_PREFIXES:
        if lowered.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    return normalized.strip().rstrip(".,;)")


def _author_name(author: Author | dict[str, Any]) -> str:
    if isinstance(author, Author):
        return author.name
    return str(author.get("name", ""))


def stable_reference_id(reference: Reference | dict[str, Any]) -> str:
    if isinstance(reference, Reference):
        title = reference.title
        year = reference.year
        authors = reference.authors
    else:
        title = str(reference.get("title", ""))
        year = reference.get("year")
        authors = reference.get("authors", [])

    author_key = "|".join(normalize_text(_author_name(author)) or "" for author in authors[:3])
    raw_key = f"{normalize_text(title) or ''}|{year or ''}|{author_key}".lower()
    digest = hashlib.sha1(raw_key.encode("utf-8")).hexdigest()[:12]
    return f"ref-{digest}"


def normalize_author(author: Author) -> Author:
    return author.model_copy(
        update={
            "name": normalize_text(author.name) or "",
            "affiliations": [
                normalized
                for affiliation in author.affiliations
                if (normalized := normalize_text(affiliation))
            ],
            "ids": {normalize_text(k) or k: normalize_text(v) or v for k, v in author.ids.items()},
        }
    )


def normalize_reference(reference: Reference) -> Reference:
    normalized = reference.model_copy(
        update={
            "reference_id": normalize_text(reference.reference_id) if reference.reference_id else None,
            "title": normalize_text(reference.title) or "",
            "venue": normalize_text(reference.venue),
            "field": normalize_text(reference.field),
            "doi": normalize_text(reference.doi),
            "raw_citation": normalize_text(reference.raw_citation),
            "authors": [normalize_author(author) for author in reference.authors],
            "external_ids": {
                normalize_text(k) or k: normalize_text(v) or v
                for k, v in reference.external_ids.items()
            },
        }
    )
    if not normalized.reference_id:
        normalized = normalized.model_copy(update={"reference_id": stable_reference_id(normalized)})
    return normalized


def normalize_citation_context(context: CitationContext) -> CitationContext:
    return context.model_copy(
        update={
            "reference_id": normalize_text(context.reference_id) or "",
            "sentence": normalize_text(context.sentence) or "",
            "paragraph": normalize_text(context.paragraph),
            "section": normalize_text(context.section),
        }
    )


def normalize_paper(paper: Paper) -> Paper:
    normalized_references = [normalize_reference(reference) for reference in paper.references]
    deduplicated_references, duplicate_reference_id_map = _deduplicate_references(
        normalized_references
    )
    citation_contexts = _remap_citation_contexts(
        paper.citation_contexts,
        duplicate_reference_id_map,
    )
    metadata = dict(paper.metadata)
    if len(deduplicated_references) < len(normalized_references):
        metadata["reference_deduplication"] = {
            "original_reference_count": len(normalized_references),
            "deduplicated_reference_count": len(deduplicated_references),
            "removed_reference_count": len(normalized_references)
            - len(deduplicated_references),
            "duplicate_reference_id_map": duplicate_reference_id_map,
        }
    return paper.model_copy(
        update={
            "paper_id": normalize_text(paper.paper_id) or "",
            "title": normalize_text(paper.title) or "",
            "abstract": normalize_text(paper.abstract),
            "venue": normalize_text(paper.venue),
            "field": normalize_text(paper.field),
            "authors": [normalize_author(author) for author in paper.authors],
            "references": deduplicated_references,
            "citation_contexts": citation_contexts,
            "metadata": metadata,
        }
    )


def _deduplicate_references(
    references: list[Reference],
) -> tuple[list[Reference], dict[str, str]]:
    deduplicated: list[Reference] = []
    index_by_key: dict[tuple[str, ...], int] = {}
    duplicate_reference_id_map: dict[str, str] = {}

    for reference in references:
        keys = _reference_dedup_keys(reference)
        existing_index = next(
            (index_by_key[key] for key in keys if key in index_by_key),
            None,
        )
        if existing_index is None:
            deduplicated.append(reference)
            kept_index = len(deduplicated) - 1
            for key in keys:
                index_by_key.setdefault(key, kept_index)
            continue

        canonical = deduplicated[existing_index]
        deduplicated[existing_index] = _merge_reference(canonical, reference)
        canonical_id = deduplicated[existing_index].reference_id
        if reference.reference_id and canonical_id and reference.reference_id != canonical_id:
            duplicate_reference_id_map[reference.reference_id] = canonical_id
        for key in keys:
            index_by_key.setdefault(key, existing_index)

    return deduplicated, duplicate_reference_id_map


def _reference_dedup_keys(reference: Reference) -> list[tuple[str, ...]]:
    keys: list[tuple[str, ...]] = []
    if doi := _reference_doi(reference):
        keys.append(("doi", doi))
    keys.extend(_reference_external_id_keys(reference))
    if title_key := _reference_title_key(reference):
        keys.append(title_key)
    if reference.reference_id:
        keys.append(("reference_id", reference.reference_id))
    return keys


def _reference_doi(reference: Reference) -> str | None:
    if doi := _canonical_doi(reference.doi):
        return doi
    for key, value in reference.external_ids.items():
        if _external_id_alias(key) == "doi" and (doi := _canonical_doi(value)):
            return doi
    return None


def _canonical_doi(value: str | None) -> str | None:
    normalized = normalize_doi(value)
    if not normalized:
        return None
    lowered = normalized.lower()
    if not lowered.startswith("10."):
        return None
    return lowered


def _reference_external_id_keys(reference: Reference) -> list[tuple[str, ...]]:
    keys = []
    for raw_key, raw_value in reference.external_ids.items():
        alias = _external_id_alias(raw_key)
        if alias is None or alias == "doi":
            continue
        value = normalize_text(raw_value)
        if not value:
            continue
        keys.append((alias, _canonical_external_id_value(alias, value)))
    return keys


def _external_id_alias(key: str) -> str | None:
    normalized = _NON_ALNUM.sub("", key.lower())
    return _EXTERNAL_ID_ALIASES.get(normalized)


def _canonical_external_id_value(alias: str, value: str) -> str:
    normalized = value.strip()
    if alias == "openalex":
        normalized = normalized.removeprefix("https://openalex.org/")
    return normalized.lower()


def _reference_title_key(reference: Reference) -> tuple[str, ...] | None:
    title = (normalize_text(reference.title) or "").lower()
    if not title:
        return None
    author_key = "|".join(
        (normalize_text(author.name) or "").lower() for author in reference.authors[:3]
    )
    year_key = str(reference.year or "")
    if len(title) < _MIN_TITLE_DEDUP_LENGTH and not (year_key and author_key):
        return None
    return ("title_year_authors", title, year_key, author_key)


def _merge_reference(canonical: Reference, duplicate: Reference) -> Reference:
    updates: dict[str, Any] = {}
    for field_name in ["title", "venue", "field", "doi", "raw_citation"]:
        canonical_value = getattr(canonical, field_name)
        duplicate_value = getattr(duplicate, field_name)
        if _missing(canonical_value) and not _missing(duplicate_value):
            updates[field_name] = duplicate_value
    if canonical.year is None and duplicate.year is not None:
        updates["year"] = duplicate.year
    if not canonical.authors and duplicate.authors:
        updates["authors"] = duplicate.authors

    external_ids = dict(duplicate.external_ids)
    external_ids.update(canonical.external_ids)
    if external_ids != canonical.external_ids:
        updates["external_ids"] = external_ids

    for key, value in _reference_extra(duplicate).items():
        if key == "deduplicated_reference_evidence":
            continue
        if key not in _reference_extra(canonical) and not _missing(value):
            updates[key] = value

    evidence = _merged_reference_evidence(canonical, duplicate)
    if len(evidence) > 1:
        updates["deduplicated_reference_evidence"] = evidence

    if not updates:
        return canonical
    return canonical.model_copy(update=updates)


def _missing(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and not value.strip()


def _reference_extra(reference: Reference) -> dict[str, Any]:
    return dict(reference.model_extra or {})


def _merged_reference_evidence(
    canonical: Reference,
    duplicate: Reference,
) -> list[dict[str, Any]]:
    existing = _reference_extra(canonical).get("deduplicated_reference_evidence")
    if isinstance(existing, list):
        evidence = [item for item in existing if isinstance(item, dict)]
    else:
        evidence = []
        if canonical_evidence := _reference_evidence(canonical):
            evidence.append(canonical_evidence)

    if duplicate_evidence := _reference_evidence(duplicate):
        if duplicate_evidence not in evidence:
            evidence.append(duplicate_evidence)
    return evidence


def _reference_evidence(reference: Reference) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for field_name in ["reference_id", "doi", "raw_citation"]:
        value = getattr(reference, field_name)
        if not _missing(value):
            evidence[field_name] = value
    if reference.external_ids:
        evidence["external_ids"] = dict(reference.external_ids)
    extra = _reference_extra(reference)
    for field_name in _REFERENCE_EVIDENCE_EXTRA_FIELDS:
        value = extra.get(field_name)
        if not _missing(value):
            evidence[field_name] = value
    return evidence


def _remap_citation_contexts(
    citation_contexts: list[CitationContext],
    duplicate_reference_id_map: dict[str, str],
) -> list[CitationContext]:
    remapped = []
    for context in citation_contexts:
        normalized_context = normalize_citation_context(context)
        reference_id = duplicate_reference_id_map.get(normalized_context.reference_id)
        if reference_id is None:
            remapped.append(normalized_context)
            continue
        remapped.append(normalized_context.model_copy(update={"reference_id": reference_id}))
    return remapped
