#!/usr/bin/env python
"""Command-line entry point for generating Q1/Q2 experimental predictions.

Argument parsing and execution are implemented in ``research/src/pipeline.py``;
this wrapper only makes that package available when run from the repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from src.pipeline import main

if __name__ == "__main__":
    # Preserve the shared pipeline implementation used by tests and experiments.
    main()
