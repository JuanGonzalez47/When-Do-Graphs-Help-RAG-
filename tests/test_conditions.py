"""Verify context construction for the RAG and graph experiment conditions."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from src.budget import count_tokens  # noqa: E402
from src.conditions import ExperimentResources, build_contexts  # noqa: E402
from src.graph import graph_from_triples  # noqa: E402
from src.retrieval import TfidfRetriever  # noqa: E402

TRIPLES = [
    ("Alpha", "causes", "Beta"),
    ("Beta", "causes", "Gamma"),
    ("Gamma", "inhibits", "Delta"),
    ("Delta", "binds", "Alpha"),
    ("Epsilon", "causes", "Alpha"),
    ("Gamma", "activates", "Epsilon"),
]

CHUNKS = [
    {"chunk_id": 0, "source": "Book", "text": "Alpha causes Beta and Beta causes Gamma."},
    {"chunk_id": 1, "source": "Book", "text": "Gamma inhibits Delta while Delta binds Alpha."},
    {"chunk_id": 2, "source": "Book", "text": "Epsilon causes Alpha and Gamma activates Epsilon."},
]


class ConditionTests(unittest.TestCase):
    """Check condition-specific context behavior and budget compliance."""

    def setUp(self) -> None:
        # Shared resources ensure each condition is compared on the same toy corpus.
        self.resources = ExperimentResources(
            retriever=TfidfRetriever().fit(CHUNKS),
            graphs={"Book": graph_from_triples(TRIPLES)},
            rag_top_k=3,
            graph_hops=2,
            seed_k=3,
        )
        self.record = {
            "id": "Novel-1",
            "source": "Book",
            "question": "What does Alpha cause?",
            "question_type": "Fact Retrieval",
            "answer": "Beta",
            "evidence": "Alpha causes Beta",
        }

    def test_rag_condition(self) -> None:
        contexts, metadata = build_contexts(self.record, "rag", 100, self.resources)
        self.assertTrue(contexts)
        self.assertLessEqual(sum(count_tokens(text) for text in contexts), 100)
        self.assertEqual(metadata["strategy"], "rag")
        self.assertEqual(metadata["condition"], "rag")
        self.assertEqual(metadata["source"], "Book")

    def test_graph_conditions(self) -> None:
        # Each graph variant must return contexts while respecting the same budget.
        for condition in ["graph_full", "graph_rewired", "graph_greedy", "graph_adaptive"]:
            contexts, metadata = build_contexts(self.record, condition, 200, self.resources)
            self.assertTrue(contexts)
            self.assertLessEqual(sum(count_tokens(text) for text in contexts), 200)
            self.assertEqual(metadata["condition"], condition)

    def test_graph_content_has_no_edges(self) -> None:
        contexts, metadata = build_contexts(self.record, "graph_content", 200, self.resources)
        self.assertTrue(contexts)
        self.assertFalse(any("--" in text for text in contexts))
        self.assertEqual(metadata["edge_count"], 0)

    def test_unknown_condition(self) -> None:
        with self.assertRaises(ValueError):
            build_contexts(self.record, "unknown", 100, self.resources)

    def test_rewired_index_cached_and_degree_preserving(self) -> None:
        first = self.resources.graph_index("Book", "graph_rewired")
        second = self.resources.graph_index("Book", "graph_rewired")
        self.assertIs(first, second)
        original = self.resources.graphs["Book"]
        # Rewiring should alter connectivity without changing the degree sequence.
        self.assertEqual(
            sorted(degree for _, degree in first.graph.degree()),
            sorted(degree for _, degree in original.degree()),
        )


if __name__ == "__main__":
    unittest.main()
