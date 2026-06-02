#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from local_impact_ranking.mineru_zip_adapter import convert_mineru_zip_to_jsonl


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
    args = parser.parse_args()

    summary = convert_mineru_zip_to_jsonl(
        args.input,
        args.output,
        with_references_output_path=args.with_references_output,
    )
    if args.summary_output:
        with open(args.summary_output, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
