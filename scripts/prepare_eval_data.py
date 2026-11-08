#!/usr/bin/env python
"""Prepare generated predictions for the official retrieval evaluator.

The benchmark evaluator expects evidence as a non-empty list of strings; this
script normalizes that field into a sibling file and leaves original predictions
unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalize_evidence(value: object) -> list[str]:
    """Convert supported evidence strings/lists to non-empty evaluator strings."""
    if isinstance(value, list):
        if not all(isinstance(item, str) and item.strip() for item in value):
            raise ValueError("Evidence lists must contain non-empty strings")
        return [item.strip() for item in value]
    if not isinstance(value, str):
        raise ValueError("Evidence must be a string or a list of strings")
    text = value.strip()
    if not text:
        return []
    if ";" in text:
        parts = [part.strip() for part in text.split(";")]
        return [part for part in parts if part]
    return [text]


def main() -> None:
    """Validate each input prediction file and write its normalized sibling."""
    parser = argparse.ArgumentParser(description="Prepare prediction files for the official evaluator")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--suffix", default="_eval")
    args = parser.parse_args()

    for path in args.inputs:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            raise ValueError(f"{path}: input must be a non-empty JSON list")
        for row in data:
            if not isinstance(row, dict):
                raise ValueError(f"{path}: each input row must be a JSON object")
            if "evidence" not in row:
                raise ValueError(f"{path}: input row is missing the evidence field")
            row["evidence"] = normalize_evidence(row.get("evidence"))
            if not row["evidence"]:
                raise ValueError(f"{path}: evidence must not be empty")
        # Write beside, but never over, the original experimental predictions.
        output = path.with_name(f"{path.stem}{args.suffix}{path.suffix}")
        output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(output)


if __name__ == "__main__":
    main()
