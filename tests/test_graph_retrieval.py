"""Verify graph-based context retrieval strategies and their reported metadata."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from src.budget import count_tokens  # noqa: E402
from src.graph import graph_from_triples  # noqa: E402
from src.graph_retrieval import GraphIndex, edge_text, query_terms  # noqa: E402

TRIPLES = [
    ("Alpha", "causes", "Beta"),
    ("Beta", "causes", "Gamma"),
    ("Gamma", "inhibits", "Delta"),
    ("Delta", "binds", "Alpha"),
    ("Epsilon", "causes", "Alpha"),
    ("Gamma", "activates", "Epsilon"),
]


class GraphRetrievalTests(unittest.TestCase):
    """Protect graph context selection under token budgets and hop limits."""

    def setUp(self) -> None:
        self.graph = graph_from_triples(TRIPLES)
        self.index = GraphIndex(self.graph, seed_k=3)

    def test_query_terms(self) -> None:
        self.assertEqual(query_terms("What causes the Alpha?"), {"causes", "alpha"})

    def test_edge_text(self) -> None:
        text = edge_text(self.graph, "alpha", "beta")
        self.assertIn("alpha", text)
        self.assertIn("causes", text)
        self.assertIn("beta", text)

    def test_full_context(self) -> None:
        result = self.index.full_context("alpha causes beta", budget=500, hops=2)
        self.assertTrue(result.contexts)
        # Full graph context serializes edges, unlike the nodes-only condition.
        self.assertTrue(all("--" in line for line in result.contexts))
        # The configured input budget applies to rendered retrieval contexts.
        self.assertLessEqual(
            sum(count_tokens(line) for line in result.contexts),
            500,
        )
        self.assertEqual(result.metadata["strategy"], "full")
        self.assertGreater(result.to_metadata()["selected_edges"], 0)

    def test_content_context_has_no_edges(self) -> None:
        result = self.index.content_context("alpha causes beta", budget=500, hops=2)
        self.assertTrue(result.contexts)
        # This isolates node content from relational structure in the comparison.
        self.assertFalse(any("--" in line for line in result.contexts))
        self.assertEqual(result.edges, [])
        self.assertEqual(result.metadata["strategy"], "content")

    def test_greedy_context(self) -> None:
        result = self.index.greedy_context("alpha causes beta gamma", budget=200, hops=2)
        self.assertTrue(result.contexts)
        self.assertLessEqual(sum(count_tokens(line) for line in result.contexts), 200)
        self.assertGreater(result.metadata["query_coverage"], 0.0)
        self.assertEqual(result.metadata["strategy"], "greedy")

    def test_adaptive_context(self) -> None:
        result = self.index.adaptive_context("alpha causes beta", budget=200, max_hops=3)
        self.assertTrue(result.contexts)
        self.assertLessEqual(sum(count_tokens(line) for line in result.contexts), 200)
        # Adaptive expansion must stay within the experiment's requested hop range.
        self.assertGreaterEqual(result.metadata["hops"], 1)
        self.assertLessEqual(result.metadata["hops"], 3)
        self.assertEqual(result.metadata["strategy"], "adaptive")

    def test_small_budget(self) -> None:
        result = self.index.full_context("alpha", budget=1)
        self.assertEqual(result.contexts, [])

    def test_empty_graph(self) -> None:
        index = GraphIndex(graph_from_triples([]))
        for result in (
            index.full_context("alpha", budget=100),
            index.content_context("alpha", budget=100),
            index.greedy_context("alpha", budget=100),
            index.adaptive_context("alpha", budget=100),
        ):
            self.assertEqual(result.contexts, [])

    def test_metadata_summary(self) -> None:
        result = self.index.full_context("alpha", budget=500)
        metadata = result.to_metadata()
        self.assertIn("selected_edges", metadata)
        self.assertEqual(metadata["contexts"], len(result.contexts))


if __name__ == "__main__":
    unittest.main()
