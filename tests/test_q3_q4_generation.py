"""Verify Q3 matched-token graph variants and the reproducible Q4 split."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import networkx as nx
import tiktoken

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_q3_q4 import (  # noqa: E402
    Q3_BUDGET,
    _make_q4_split,
    _q3_graph_variants,
    _rendered_tokens,
    _rewire_selected_edges,
)
from src.graph_retrieval import GraphIndex  # noqa: E402
from src.data import LEVELS  # noqa: E402


class Q3Q4GenerationTests(unittest.TestCase):
    """Protect controls central to interpreting the Q3 and Q4 comparisons."""

    def test_selected_edge_rewiring_preserves_degree_and_edge_count(self) -> None:
        edges = [
            ("a", "b"),
            ("b", "c"),
            ("c", "d"),
            ("d", "e"),
            ("e", "f"),
            ("f", "a"),
            ("a", "d"),
            ("b", "e"),
        ]
        rewired, swaps = _rewire_selected_edges(edges, seed=13)
        original_graph = nx.Graph(edges)
        rewired_graph = nx.Graph(rewired)
        self.assertGreater(swaps, 0)
        self.assertEqual(len(edges), len(rewired))
        self.assertEqual(
            sorted(dict(original_graph.degree()).values()),
            sorted(dict(rewired_graph.degree()).values()),
        )

    def test_q3_graph_variants_have_exactly_matched_serialized_tokens(self) -> None:
        graph = nx.Graph()
        triples = [
            ("Alice", "knows", "Bob"),
            ("Bob", "visited", "Paris"),
            ("Paris", "contains", "Museum"),
            ("Museum", "displays", "Painting"),
            ("Painting", "created by", "Alice"),
            ("Bob", "works with", "Carol"),
            ("Carol", "lives in", "Paris"),
            ("Museum", "located in", "Paris"),
        ]
        for head, relation, tail in triples:
            graph.add_edge(head.lower(), tail.lower(), relations=[relation])
        encoding = tiktoken.get_encoding("cl100k_base")
        variants = _q3_graph_variants(
            GraphIndex(graph, seed_k=10),
            "Who knows Bob and where is the museum?",
            Q3_BUDGET,
            "Novel-test-id",
            encoding,
        )
        # Matching rendered-token counts controls context size across Q3 graph variants.
        self.assertEqual(
            {
                _rendered_tokens(contexts, encoding)
                for contexts, _ in variants.values()
            },
            {next(iter(variants.values()))[1]["matched_context_tokens"]},
        )
        self.assertLessEqual(
            next(iter(variants.values()))[1]["matched_context_tokens"],
            Q3_BUDGET,
        )
        self.assertGreater(
            variants["graph_rewired_topology"][1]["rewire_swaps"],
            0,
        )
        self.assertFalse(any("--" in line for line in variants["graph_nodes_only"][0][0].splitlines()))

    def test_q4_split_is_reproducible_and_stratified(self) -> None:
        records = []
        for split, prefix in (("novel", "Novel"), ("medical", "Medical")):
            for level_index, level in enumerate(LEVELS):
                for sample_index in range(50):
                    records.append(
                        {
                            "id": f"{prefix}-{level_index}-{sample_index}",
                            "question_type": level,
                        }
                    )
        first = _make_q4_split(records)
        second = _make_q4_split(records)
        # Repeated split generation must yield identical, non-overlapping partitions.
        self.assertEqual(first, second)
        self.assertEqual(len(first["development_ids"]), 320)
        self.assertEqual(len(first["confirmation_ids"]), 80)
        self.assertFalse(
            set(first["development_ids"]) & set(first["confirmation_ids"])
        )


if __name__ == "__main__":
    unittest.main()
