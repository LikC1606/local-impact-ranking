from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Author(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    affiliations: list[str] = Field(default_factory=list)
    ids: dict[str, str] = Field(default_factory=dict)


class Reference(BaseModel):
    model_config = ConfigDict(extra="allow")

    reference_id: str | None = None
    title: str
    authors: list[Author] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    field: str | None = None
    doi: str | None = None
    external_ids: dict[str, str] = Field(default_factory=dict)
    raw_citation: str | None = None


class CitationContext(BaseModel):
    model_config = ConfigDict(extra="allow")

    reference_id: str
    sentence: str
    paragraph: str | None = None
    section: str | None = None


class Paper(BaseModel):
    model_config = ConfigDict(extra="allow")

    paper_id: str
    title: str
    references: list[Reference]
    abstract: str | None = None
    year: int | None = None
    venue: str | None = None
    field: str | None = None
    authors: list[Author] = Field(default_factory=list)
    citation_contexts: list[CitationContext] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
