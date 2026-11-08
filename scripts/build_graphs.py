#!/usr/bin/env python
"""Command-line entry point for extracting and saving per-source knowledge graphs.

The implementation lives in ``research/src/graph_extraction.py`` so it can also
be imported and tested independently of this executable wrapper.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from src.graph_extraction import main

if __name__ == "__main__":
    # Keep command-line behavior delegated to the tested research implementation.
    main()
