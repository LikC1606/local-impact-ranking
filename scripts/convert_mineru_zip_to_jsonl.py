#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from local_impact_ranking.mineru_zip_adapter import (
    DEFAULT_MAX_COMPRESSION_RATIO,
    DEFAULT_MAX_MARKDOWN_BYTES,
    DEFAULT_MAX_MARKDOWN_FILES,
    DEFAULT_MAX_TOTAL_MARKDOWN_BYTES,
    convert_mineru_zip_to_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a MinerU-style paper ZIP into local impact JSONL.",
    )
    parser.add_argument("--input", required=True, help="Input ZIP path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--with-references-output",
        help="Optional JSONL output containing only records with extracted references.",
    )
    parser.add_argument(
        "--summary-output",
        help="Optional JSON summary path.",
    )
    parser.add_argument(
        "--max-markdown-bytes",
        type=int,
        default=DEFAULT_MAX_MARKDOWN_BYTES,
        help="Maximum uncompressed bytes per markdown entry.",
    )
    parser.add_argument(
        "--max-total-markdown-bytes",
        type=int,
        default=DEFAULT_MAX_TOTAL_MARKDOWN_BYTES,
        help="Maximum total uncompressed markdown bytes in the ZIP.",
    )
    parser.add_argument(
        "--max-markdown-files",
        type=int,
        default=DEFAULT_MAX_MARKDOWN_FILES,
        help="Maximum MinerU markdown entries to convert.",
    )
    parser.add_argument(
        "--max-compression-ratio",
        type=float,
        default=DEFAULT_MAX_COMPRESSION_RATIO,
        help="Maximum allowed uncompressed/compressed ratio for markdown entries.",
    )
    args = parser.parse_args()

    summary = convert_mineru_zip_to_jsonl(
        args.input,
        args.output,
        with_references_output_path=args.with_references_output,
        max_markdown_bytes=args.max_markdown_bytes,
        max_total_markdown_bytes=args.max_total_markdown_bytes,
        max_markdown_files=args.max_markdown_files,
        max_compression_ratio=args.max_compression_ratio,
    )
    if args.summary_output:
        Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_output, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
