"""Verify LLM triple extraction, validation, caching, and source graph building."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from src import graph_extraction  # noqa: E402
from src.graph import load_graph  # noqa: E402
from src.graph_extraction import (  # noqa: E402
    build_source_graph,
    dedupe_triples,
    extract_chunk,
    validate_triples,
)


class FakeClient:
    def __init__(self, response) -> None:
        self.response = response
        self.calls = 0

    def complete_json(self, system: str, user: str, **kwargs):
        self.calls += 1
        return self.response


class GraphExtractionTests(unittest.TestCase):
    """Protect the conversion of corpus chunks into reusable source graphs."""

    def test_validate_triples(self) -> None:
        items = [
            {"head": "A", "relation": "r", "tail": "B"},
            {"head": "A", "relation": "r", "tail": "A"},
            {"head": "", "relation": "r", "tail": "B"},
            {"bad": "shape"},
        ]
        self.assertEqual(validate_triples(items), [("A", "r", "B")])
        self.assertEqual(validate_triples({"triples": items}), [("A", "r", "B")])
        self.assertEqual(validate_triples("not a list"), [])

    def test_validate_triples_drops_metadata(self) -> None:
        items = [
            {"head": "Corpus 10762", "relation": "contains", "tail": "chunk id 3"},
            {"head": "Novel 10762", "relation": "contains", "tail": "Plot"},
            {"head": "Jane", "relation": "wrote", "tail": "Novel-10762"},
            {"head": "Project Gutenberg", "relation": "produced", "tail": "Book"},
            {"head": "198", "relation": "count", "tail": "pages"},
            {"head": "A", "relation": "r", "tail": "B"},
        ]
        self.assertEqual(validate_triples(items, source="Novel-10762"), [("A", "r", "B")])
        self.assertEqual(
            validate_triples(
                [{"head": "Corpus luteum", "relation": "secretes", "tail": "progesterone"}],
                source="Medical",
            ),
            [("Corpus luteum", "secretes", "progesterone")],
        )

    def test_dedupe_triples(self) -> None:
        triples = [("The A", "r", "B"), ("a", "r", "b"), ("C", "r", "D")]
        self.assertEqual(dedupe_triples(triples), [("The A", "r", "B"), ("C", "r", "D")])

    def test_extract_chunk_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(graph_extraction, "TRIPLE_CACHE", Path(tmp)):
                client = FakeClient([{"head": "A", "relation": "r", "tail": "B"}])
                chunk = {"chunk_id": 0, "text": "A relates to B."}
                first = extract_chunk("novel", "Book-1", chunk, client)
                second = extract_chunk("novel", "Book-1", chunk, client)
                self.assertEqual(first, [("A", "r", "B")])
                self.assertEqual(second, first)
                # The second identical extraction should reuse the cache, not call the LLM.
                self.assertEqual(client.calls, 1)

    def test_build_source_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(graph_extraction, "CHUNK_CACHE", root / "chunks"), mock.patch.object(
                graph_extraction, "TRIPLE_CACHE", root / "extraction"
            ), mock.patch.object(graph_extraction, "GRAPHS", root / "graphs"):
                client = FakeClient(
                    [
                        {"head": "Megalosaurus", "relation": "interpreted", "tail": "Grampus"},
                        {"head": "Grampus", "relation": "criticized", "tail": "Megalosaurus"},
                    ]
                )
                summary = build_source_graph(
                    "novel",
                    "Novel-10762",
                    client,
                    workers=1,
                    limit_chunks=1,
                )
                # A one-chunk smoke build must persist a loadable source graph.
                self.assertEqual(summary["chunks"], 1)
                self.assertGreater(summary["triples"], 0)
                graph = load_graph(summary["path"])
                self.assertEqual(graph.number_of_nodes(), 2)
                self.assertGreater(graph.number_of_edges(), 0)


if __name__ == "__main__":
    unittest.main()
