from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import median
from typing import Any
from zipfile import ZipFile, ZipInfo


DEFAULT_MAX_MARKDOWN_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_TOTAL_MARKDOWN_BYTES = 50 * 1024 * 1024 * 1024
DEFAULT_MAX_MARKDOWN_FILES = 200_000
DEFAULT_MAX_COMPRESSION_RATIO = 1000.0

DOI_RE = re.compile(
    r"(?:doi\s*[: ]\s*|https?://doi\.org/)?(10\.\d{4,9}/[^\s,;\]\)]+)",
    re.I,
)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
REF_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?\s*"
    r"(references|reference|bibliography|literature cited)"
    r"(?:\s+and\s+notes)?\s*(?:\*\*)?\s*:?\s*$",
    re.I,
)
REFERENCE_START_RE = re.compile(
    r"^\s*(?:\[(?:S)?\d+\]|\((?:S)?\d+\)|(?:S)?\d+\.?)\s+",
)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
BAD_TITLE_RE = re.compile(
    r"^(open access|citation|copyright|keywords?|abstract|introduction|"
    r"references?|supplemental information|supplementary information|"
    r"supplementary material|supplemental material|supplemental data|"
    r"supplementary data|paper|article|review|check for updates)$",
    re.I,
)
ADMIN_RE = re.compile(
    r"(publisher.?s note|generative ai statement|data availability|"
    r"author contributions|conflict of interest|ethics statement|funding|"
    r"acknowledg(e)?ments)",
    re.I,
)


def convert_mineru_zip_to_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    with_references_output_path: str | Path | None = None,
    max_markdown_bytes: int = DEFAULT_MAX_MARKDOWN_BYTES,
    max_total_markdown_bytes: int = DEFAULT_MAX_TOTAL_MARKDOWN_BYTES,
    max_markdown_files: int = DEFAULT_MAX_MARKDOWN_FILES,
    max_compression_ratio: float = DEFAULT_MAX_COMPRESSION_RATIO,
) -> dict[str, Any]:
    records = []
    source = Path(input_path)
    with ZipFile(source) as archive:
        markdown_infos = _validated_markdown_infos(
            archive,
            max_markdown_bytes=max_markdown_bytes,
            max_total_markdown_bytes=max_total_markdown_bytes,
            max_markdown_files=max_markdown_files,
            max_compression_ratio=max_compression_ratio,
        )
        for markdown_info in markdown_infos:
            markdown_name = markdown_info.filename
            paper_id = markdown_name.split("/", 1)[0]
            text = _read_markdown_text(
                archive,
                markdown_info,
                max_markdown_bytes=max_markdown_bytes,
            )
            records.append(_record_from_markdown(paper_id, markdown_name, text, source))

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(records, destination)
    if with_references_output_path is not None:
        _write_jsonl(
            [record for record in records if record["references"]],
            Path(with_references_output_path),
        )

    reference_counts = [len(record["references"]) for record in records]
    without_references = [
        record["paper_id"] for record in records if not record["references"]
    ]
    return {
        "source_zip": str(source),
        "paper_count": len(records),
        "with_references_count": len(records) - len(without_references),
        "without_references_count": len(without_references),
        "abstract_count": sum(1 for record in records if record.get("abstract")),
        "doi_count": sum(1 for record in records if record.get("doi")),
        "reference_count": {
            "min": min(reference_counts) if reference_counts else 0,
            "median": median(reference_counts) if reference_counts else 0,
            "max": max(reference_counts) if reference_counts else 0,
            "total": sum(reference_counts),
        },
        "without_references_paper_ids": without_references,
        "output_path": str(destination),
        "with_references_output_path": (
            str(with_references_output_path) if with_references_output_path else None
        ),
    }


def _validated_markdown_infos(
    archive: ZipFile,
    *,
    max_markdown_bytes: int,
    max_total_markdown_bytes: int,
    max_markdown_files: int,
    max_compression_ratio: float,
) -> list[ZipInfo]:
    markdown_infos = [
        info
        for info in archive.infolist()
        if re.match(r"^([^/]+)/auto/\1\.md$", info.filename)
    ]
    if len(markdown_infos) > max_markdown_files:
        raise ValueError(
            f"MinerU ZIP contains {len(markdown_infos)} markdown files, "
            f"exceeds max_markdown_files={max_markdown_files}"
        )

    total_bytes = 0
    for info in markdown_infos:
        if info.file_size > max_markdown_bytes:
            raise ValueError(
                f"{info.filename} is {info.file_size} bytes, "
                f"exceeds max_markdown_bytes={max_markdown_bytes}"
            )
        total_bytes += info.file_size
        if total_bytes > max_total_markdown_bytes:
            raise ValueError(
                f"MinerU ZIP markdown total is {total_bytes} bytes, "
                f"exceeds max_total_markdown_bytes={max_total_markdown_bytes}"
            )
        if info.file_size > 0 and info.compress_size == 0:
            raise ValueError(f"{info.filename} has zero compressed size")
        if info.compress_size > 0:
            ratio = info.file_size / info.compress_size
            if ratio > max_compression_ratio:
                raise ValueError(
                    f"{info.filename} compression ratio {ratio:.1f} "
                    f"exceeds max_compression_ratio={max_compression_ratio}"
                )
    return sorted(markdown_infos, key=lambda info: info.filename)


def _read_markdown_text(
    archive: ZipFile,
    info: ZipInfo,
    *,
    max_markdown_bytes: int,
) -> str:
    chunks: list[bytes] = []
    total = 0
    with archive.open(info) as handle:
        while True:
            chunk = handle.read(min(64 * 1024, max_markdown_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_markdown_bytes:
                raise ValueError(
                    f"{info.filename} exceeds max_markdown_bytes={max_markdown_bytes}"
                )
    return b"".join(chunks).decode("utf-8", "replace")


def _record_from_markdown(
    paper_id: str,
    markdown_name: str,
    text: str,
    source: Path,
) -> dict[str, Any]:
    title = _extract_title(text)
    doi = _extract_doi(_target_text_before_references(text))
    references = _extract_references(text, paper_id)
    metadata = {
        "source_zip": str(source),
        "markdown_path": markdown_name,
        "markdown_char_count": len(text),
        "extracted_reference_count": len(references),
        "source_kind": _source_kind(text, title),
        "extraction_status": (
            "references_extracted" if references else "no_references_extracted"
        ),
    }
    if doi:
        metadata["external_ids"] = {"doi": doi}
    record: dict[str, Any] = {
        "paper_id": paper_id,
        "title": title,
        "references": references,
        "metadata": metadata,
    }
    abstract = _extract_abstract(text, title)
    if abstract:
        record["abstract"] = abstract
    year = _extract_year(text, doi)
    if year:
        record["year"] = year
    if doi:
        record["doi"] = doi
    return record


def _extract_title(text: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines[:260]):
        match = HEADING_RE.match(line.strip())
        if match is None:
            continue
        title = _clean(match.group(2))
        if _is_bad_title(title):
            continue
        parts = [title]
        cursor = index + 1
        while cursor < min(len(lines), index + 8):
            raw = lines[cursor].strip()
            if not raw:
                cursor += 1
                continue
            heading = HEADING_RE.match(raw)
            if heading is not None:
                continuation = _clean(heading.group(2))
                if _is_bad_title(continuation):
                    break
                if len(" ".join(parts)) < 110 or continuation[:1].islower():
                    parts.append(continuation)
                    cursor += 1
                    continue
            elif raw[:1].islower() and len(" ".join(parts)) < 130:
                parts.append(_clean(raw))
                cursor += 1
                continue
            break
        return _clean(" ".join(parts))[:500]
    for line in lines[:160]:
        title = _clean(line)
        if not _is_bad_title(title):
            return title[:500]
    return "Untitled paper"


def _extract_abstract(text: str, title: str) -> str | None:
    explicit = re.search(
        r"^#{1,6}\s*(?:Abstract|Summary)\s*\n(.*?)(?=\n#{1,6}\s+|\Z)",
        text,
        re.I | re.M | re.S,
    )
    if explicit:
        abstract = _clean(explicit.group(1))
        if len(abstract) > 40:
            return abstract[:2500]
    end = re.search(r"\n#{1,4}\s*(KEYWORDS?|1\s+Introduction|Introduction)\b", text, re.I)
    prefix = text[: end.start()] if end else text[:7000]
    marker = re.search(
        r"\b(Background|Objective|Purpose|Importance|Methods|Results|Conclusion)\s*:",
        prefix,
        re.I,
    )
    if marker:
        abstract = _clean(prefix[marker.start() :])
        if len(abstract) > 80:
            return abstract[:2500]
    title_position = text.find(title)
    if title_position >= 0:
        candidate = _clean(text[title_position + len(title) : title_position + 2200])
        if len(candidate) > 80:
            return candidate[:2500]
    return None


def _extract_references(text: str, paper_id: str) -> list[dict[str, Any]]:
    entries = _reference_entries_after_heading(text)
    if not entries:
        entries = _numbered_reference_entries(text)

    references = []
    seen: set[tuple[str, str]] = set()
    for raw in entries:
        if len(raw) < 30 or ADMIN_RE.search(raw):
            continue
        if not (YEAR_RE.search(raw) or DOI_RE.search(raw)):
            continue
        title = _reference_title(raw)
        doi = _extract_doi(raw)
        year_match = YEAR_RE.search(raw)
        key = _reference_dedup_key(
            title,
            doi,
            year_match.group(1) if year_match else None,
        )
        if key in seen:
            continue
        seen.add(key)
        reference = {
            "reference_id": f"{paper_id}_ref_{len(references) + 1:03d}",
            "title": title,
            "raw_citation": raw[:2000],
        }
        if doi:
            reference["doi"] = doi
            reference["external_ids"] = {"doi": doi}
        if year_match:
            reference["year"] = int(year_match.group(1))
        references.append(reference)
    return references


def _reference_dedup_key(
    title: str,
    doi: str | None,
    year: str | None,
) -> tuple[str, str]:
    if doi:
        return ("doi", doi)
    normalized_title = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
    return ("title_year", f"{normalized_title}|{year or ''}")


def _reference_entries_after_heading(text: str) -> list[str]:
    lines = text.splitlines()
    start_index = None
    for index, line in enumerate(lines):
        if REF_HEADING_RE.match(line.strip()):
            start_index = index + 1
            break
    if start_index is None:
        return []
    entries = _reference_entries_from_lines(lines[start_index:])
    if entries:
        return entries
    return _reference_paragraphs_from_lines(lines[start_index:])


def _numbered_reference_entries(text: str) -> list[str]:
    lines = text.splitlines()
    blocks = _reference_entries_from_lines(lines)
    if len(blocks) >= 2:
        return blocks
    if blocks and _has_single_reference_tail(lines):
        return blocks
    return []


def _target_text_before_references(text: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if REF_HEADING_RE.match(line.strip()):
            return "\n".join(lines[:index])
    for index, line in enumerate(lines):
        stripped = _clean(line)
        if not REFERENCE_START_RE.match(stripped):
            continue
        if not (YEAR_RE.search(stripped) or DOI_RE.search(stripped)):
            continue
        following_reference_count = 0
        for candidate in lines[index : index + 40]:
            candidate_text = _clean(candidate)
            if REFERENCE_START_RE.match(candidate_text) and (
                YEAR_RE.search(candidate_text) or DOI_RE.search(candidate_text)
            ):
                following_reference_count += 1
        if following_reference_count >= 2 or _looks_like_single_reference_tail(
            lines,
            index,
            following_reference_count,
        ):
            return "\n".join(lines[:index])
    return text


def _has_single_reference_tail(lines: list[str]) -> bool:
    for index, line in enumerate(lines):
        stripped = _clean(line)
        if not REFERENCE_START_RE.match(stripped):
            continue
        if not (YEAR_RE.search(stripped) or DOI_RE.search(stripped)):
            continue
        following_reference_count = sum(
            1
            for candidate in lines[index : index + 40]
            if REFERENCE_START_RE.match(_clean(candidate))
            and (YEAR_RE.search(_clean(candidate)) or DOI_RE.search(_clean(candidate)))
        )
        if _looks_like_single_reference_tail(lines, index, following_reference_count):
            return True
    return False


def _looks_like_single_reference_tail(
    lines: list[str],
    index: int,
    following_reference_count: int,
) -> bool:
    if following_reference_count != 1:
        return False
    nonempty_after = [_clean(line) for line in lines[index:] if _clean(line)]
    if len(nonempty_after) > 6:
        return False
    return index >= max(3, len(lines) // 3)


def _reference_paragraphs_from_lines(lines: list[str]) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        raw = line.strip()
        heading = HEADING_RE.match(raw)
        if heading is not None and not REF_HEADING_RE.match(raw):
            heading_text = _clean(heading.group(2))
            if ADMIN_RE.search(heading_text):
                continue
        stripped = _clean(raw)
        if not stripped or ADMIN_RE.search(stripped):
            if current:
                paragraphs.append(_clean(" ".join(current)))
                current = []
            continue
        if stripped.startswith("![]("):
            continue
        current.append(stripped)
    if current:
        paragraphs.append(_clean(" ".join(current)))
    return [
        paragraph
        for paragraph in paragraphs
        if len(paragraph) >= 30 and (YEAR_RE.search(paragraph) or DOI_RE.search(paragraph))
    ]


def _reference_entries_from_lines(lines: list[str]) -> list[str]:
    entries: list[str] = []
    current: list[str] = []
    for line in lines:
        stripped = _clean(line)
        if not stripped or stripped.startswith("![]("):
            continue
        heading = HEADING_RE.match(line.strip())
        if heading is not None and not REF_HEADING_RE.match(line.strip()):
            heading_text = _clean(heading.group(2))
            if ADMIN_RE.search(heading_text):
                continue
        if REFERENCE_START_RE.match(stripped):
            if current:
                entries.append(_clean(" ".join(current)))
            current = [stripped]
            continue
        if current:
            if _looks_like_non_reference_continuation(stripped):
                continue
            current.append(stripped)
    if current:
        entries.append(_clean(" ".join(current)))
    return entries


def _looks_like_non_reference_continuation(text: str) -> bool:
    return bool(re.match(r"^(figure|fig\.|table|scheme)\s+", text, re.I))


def _reference_title(raw: str) -> str:
    text = re.sub(
        r"^\s*(?:\[(?:S)?\d+\]|\((?:S)?\d+\)|(?:S)?\d+\.?)\s+",
        "",
        raw,
    ).strip()
    acs_title = re.search(
        r"^(?:[A-Z][^.;]{0,70},\s*(?:[A-Z]\.\s*)+;?\s*)+"
        r"([A-Z][^.]{15,}?)\.\s+(?:[A-Z][A-Za-z.& ]{2,}|\d|https?://|doi\b)",
        text,
    )
    if acs_title:
        return _clean(acs_title.group(1)).strip(". ")
    comma_title = re.search(
        r"(?:^|,\s)([A-Z][^.]{15,}?)\.\s+(?:[A-Z][A-Za-z.& ]{2,}|\d|https?://|doi\b)",
        text,
    )
    if comma_title:
        return _clean(comma_title.group(1)).strip(". ")
    author_year = re.search(r"\((?:19|20)\d{2}[a-z]?\)\.\s*(.+)", text)
    if author_year:
        title_part = author_year.group(1)
        return _first_sentence_title(title_part)
    return _first_sentence_title(text)


def _first_sentence_title(text: str) -> str:
    text = re.sub(r"\bdoi\s*[: ]\s*10\.\S+.*$", "", text, flags=re.I)
    text = re.sub(r"https?://doi\.org/10\.\S+.*$", "", text, flags=re.I)
    pieces = re.split(
        r"\.\s+(?=[A-Z][A-Za-z& ]{2,}\b|\d+\s*[,(]|https?://|doi\b)",
        text,
        maxsplit=1,
    )
    candidate = _clean(pieces[0]).strip(". ")
    if len(candidate) > 260:
        candidate = candidate[:260].rsplit(" ", 1)[0]
    return candidate or _clean(text[:180]).strip(". ")


def _extract_doi(text: str) -> str | None:
    match = DOI_RE.search(text)
    if match is None:
        return None
    return match.group(1).rstrip(".").rstrip(")").rstrip("]").lower()


def _extract_year(text: str, doi: str | None) -> int | None:
    for pattern in [
        r"PUBLISHED\s+\d{1,2}\s+[A-Za-z]+\s+(\d{4})",
        r"published\s+online\s+\d{1,2}\s+[A-Za-z]+\s+(\d{4})",
        r"accepted\s+\w+\s+\d{1,2},?\s+(\d{4})",
        r"accepted\s+\d{1,2}\s+[A-Za-z]+\s+(\d{4})",
        r"Cite this:[^\n]{0,120}\b(20\d{2})\b",
        r"©\s*(20\d{2})\b",
        r"received\s+\w+\s+\d{1,2},?\s+(\d{4})",
        r"received\s+\d{1,2}\s+[A-Za-z]+\s+(\d{4})",
    ]:
        match = re.search(pattern, text[:8000], re.I)
        if match:
            return int(match.group(1))
    if doi:
        match = re.search(r"\b(20\d{2})\b", doi)
        if match:
            return int(match.group(1))
    return None


def _source_kind(text: str, title: str) -> str:
    candidates = [title]
    first_heading = _first_heading_text(text)
    if first_heading:
        candidates.append(first_heading)
    if any(_looks_like_supplement_title(candidate) for candidate in candidates):
        return "supplement_or_extracted_supplement"
    return "article_or_extracted_article"


def _first_heading_text(text: str) -> str | None:
    for line in text.splitlines()[:80]:
        match = HEADING_RE.match(line.strip())
        if match:
            return _clean(match.group(2))
    return None


def _looks_like_supplement_title(text: str) -> bool:
    normalized = text.lower().strip(": .")
    if re.match(r"^(fig(?:ure)?\s*s?\d+|figs\d+|table\s*s?\d+)\b", normalized):
        return True
    return normalized.startswith(
        (
            "supplementary",
            "supplemental",
            "supporting information",
            "supporting material",
            "supplementary material",
            "supplemental material",
            "supplementary file",
            "supplemental file",
            "supplementary table",
            "supplemental table",
            "supplementary figure",
            "supplemental figure",
        )
    )


def _is_bad_title(text: str) -> bool:
    value = _clean(text).strip(": .")
    if not value or len(value) < 8:
        return True
    if BAD_TITLE_RE.match(value):
        return True
    if value.startswith("http"):
        return True
    return False


def _clean(text: str) -> str:
    text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("\u00ad", "")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n#*")


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
