"""Verify serialization and stable cache-key helpers used by experiment runs."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from src.io import (  # noqa: E402
    append_jsonl,
    cache_key,
    read_json,
    read_jsonl,
    read_yaml,
    sha256_text,
    write_json,
    write_jsonl,
    write_yaml,
)


class IOTests(unittest.TestCase):
    """Check persistence primitives used for configs, predictions, and caches."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_json_roundtrip(self) -> None:
        path = self.root / "nested" / "data.json"
        write_json(path, {"a": [1, 2, 3], "b": "café"})
        self.assertEqual(read_json(path), {"a": [1, 2, 3], "b": "café"})

    def test_jsonl_roundtrip(self) -> None:
        path = self.root / "rows.jsonl"
        rows = [{"id": "a"}, {"id": "b"}]
        write_jsonl(path, rows)
        self.assertEqual(read_jsonl(path), rows)
        append_jsonl(path, {"id": "c"})
        self.assertEqual(read_jsonl(path), rows + [{"id": "c"}])
        self.assertEqual(read_jsonl(self.root / "missing.jsonl"), [])

    def test_yaml_roundtrip(self) -> None:
        path = self.root / "config.yaml"
        data = {"seed": 42, "levels": ["a", "b"], "nested": {"x": 1}}
        write_yaml(path, data)
        self.assertEqual(read_yaml(path), data)

    def test_hashing(self) -> None:
        # Stable hashes identify cached requests; changing an input changes its key.
        self.assertEqual(sha256_text("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        self.assertEqual(cache_key("a", "b"), cache_key("a", "b"))
        self.assertNotEqual(cache_key("a", "b"), cache_key("a", "c"))


if __name__ == "__main__":
    unittest.main()
