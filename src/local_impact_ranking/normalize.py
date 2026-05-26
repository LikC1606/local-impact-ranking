from __future__ import annotations

import hashlib
import re
from typing import Any

from local_impact_ranking.models import Author, Paper, Reference

_WHITESPACE = re.compile(r"\s+")


def normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    return _WHITESPACE.sub(" ", value).strip()


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


def normalize_paper(paper: Paper) -> Paper:
    return paper.model_copy(
        update={
            "paper_id": normalize_text(paper.paper_id) or "",
            "title": normalize_text(paper.title) or "",
            "abstract": normalize_text(paper.abstract),
            "venue": normalize_text(paper.venue),
            "field": normalize_text(paper.field),
            "authors": [normalize_author(author) for author in paper.authors],
            "references": [normalize_reference(reference) for reference in paper.references],
        }
    )
