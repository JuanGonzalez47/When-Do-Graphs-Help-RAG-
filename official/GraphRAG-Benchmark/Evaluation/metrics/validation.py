from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable


def normalize_statement(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def require_exact_statements(expected: Iterable[str], actual: Iterable[Any], label: str) -> None:
    expected_values = [normalize_statement(value) for value in expected]
    actual_values = [
        normalize_statement(value) if isinstance(value, str) else ""
        for value in actual
    ]
    if any(not value for value in actual_values):
        raise ValueError(f"{label}: classifications must contain non-empty statement strings")
    if Counter(expected_values) != Counter(actual_values):
        raise ValueError(
            f"{label}: expected exactly {len(expected_values)} classifications; "
            f"received {len(actual_values)} with missing, duplicated, or unexpected statements"
        )
