"""Verify graph construction, experiment variants, and graph-quality summaries."""

from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

import networkx as nx  # noqa: E402

from src.graph import (  # noqa: E402
    evidence_coverage,
    graph_from_triples,
    graph_metrics,
    graph_to_records,
    normalize_entity,
    parse_evidence_triple,
    prune_edges_centrality,
    prune_edges_random,
    reachability,
    remove_edges,
    rewire_edges,
    subgraph_by_edges,
)


class GraphTests(unittest.TestCase):
    """Check graph operations that underpin graph conditions and diagnostics."""

    def setUp(self) -> None:
        self.triples = [
            ("Alpha", "is related to", "Beta"),
            ("Beta", "causes", "Gamma"),
            ("Gamma", "is related to", "Alpha"),
            ("Delta", "causes", "Gamma"),
        ]

    def test_normalize_entity(self) -> None:
        self.assertEqual(normalize_entity("The  Cat!"), "cat")
        self.assertEqual(normalize_entity("A Tale of Two Cities"), "tale of two cities")
        self.assertEqual(normalize_entity("  "), "")

    def test_parse_evidence_triple(self) -> None:
        self.assertEqual(
            parse_evidence_triple("(erica vagans, is also known as, Cornish heath)."),
            ("erica vagans", "is also known as", "Cornish heath"),
        )
        self.assertIsNone(parse_evidence_triple("not a triple"))

    def test_graph_from_triples(self) -> None:
        graph = graph_from_triples(self.triples)
        self.assertEqual(graph.number_of_nodes(), 4)
        self.assertEqual(graph.number_of_edges(), 4)
        self.assertIn("causes", graph["beta"]["gamma"]["relations"])
        self.assertEqual(graph["beta"]["gamma"]["weight"], 1)

    def test_graph_from_triples_aggregates_relations(self) -> None:
        graph = graph_from_triples(self.triples + [("Alpha", "causes", "Beta")])
        self.assertEqual(sorted(graph["alpha"]["beta"]["relations"]), ["causes", "is related to"])
        self.assertEqual(graph["alpha"]["beta"]["weight"], 2)

    def test_remove_edges_keeps_nodes(self) -> None:
        graph = graph_from_triples(self.triples)
        content = remove_edges(graph)
        self.assertEqual(content.number_of_nodes(), graph.number_of_nodes())
        self.assertEqual(content.number_of_edges(), 0)
        self.assertTrue(all(degree == 0 for _, degree in content.degree()))

    def test_rewire_preserves_degrees(self) -> None:
        graph = nx.cycle_graph(12)
        rewired = rewire_edges(graph, seed=7, swap_fraction=0.5)
        self.assertEqual(set(rewired.nodes()), set(graph.nodes()))
        # Degree preservation is a structural control, not proof of preserved edge attributes.
        self.assertEqual(
            sorted(degree for _, degree in rewired.degree()),
            sorted(degree for _, degree in graph.degree()),
        )

    def test_rewire_small_graph_noop(self) -> None:
        graph = nx.path_graph(3)
        rewired = rewire_edges(graph)
        self.assertEqual(sorted(rewired.edges()), sorted(graph.edges()))

    def test_subgraph_by_edges(self) -> None:
        graph = graph_from_triples(self.triples)
        selected = subgraph_by_edges(graph, [("alpha", "beta"), ("missing", "edge")])
        self.assertEqual(selected.number_of_edges(), 1)
        self.assertEqual(set(selected.nodes()), {"alpha", "beta"})

    def test_prune_random(self) -> None:
        graph = nx.path_graph(11)
        pruned = prune_edges_random(graph, keep_fraction=0.5, seed=1)
        self.assertEqual(pruned.number_of_edges(), 5)
        again = prune_edges_random(graph, keep_fraction=0.5, seed=1)
        self.assertEqual(sorted(pruned.edges()), sorted(again.edges()))

    def test_prune_centrality(self) -> None:
        graph = nx.path_graph(5)
        pruned = prune_edges_centrality(graph, keep_fraction=0.5)
        self.assertEqual(pruned.number_of_edges(), 2)

    def test_metrics(self) -> None:
        graph = nx.path_graph(3)
        metrics = graph_metrics(graph)
        self.assertEqual(metrics["node_count"], 3)
        self.assertEqual(metrics["edge_count"], 2)
        self.assertAlmostEqual(metrics["average_degree"], 4 / 3)
        self.assertAlmostEqual(metrics["average_clustering"], 0.0)
        self.assertEqual(metrics["components"], 1)
        self.assertEqual(metrics["isolated_nodes"], 0)

    def test_evidence_coverage(self) -> None:
        graph = graph_from_triples(self.triples)
        coverage = evidence_coverage(graph, [self.triples[0], ("Alpha", "unknown", "Gamma")])
        self.assertEqual(coverage["triples"], 2)
        self.assertAlmostEqual(coverage["structural_coverage"], 1.0)
        strict = evidence_coverage(
            graph,
            [self.triples[0], ("Alpha", "unknown", "Gamma")],
            require_relation=True,
        )
        self.assertAlmostEqual(strict["relation_coverage"], 0.5)

    def test_reachability(self) -> None:
        graph = graph_from_triples(self.triples)
        self.assertAlmostEqual(reachability(graph, ["Alpha"], ["Gamma", "Delta"]), 1.0)
        disconnected = graph.copy()
        disconnected.add_node("isolated")
        self.assertAlmostEqual(reachability(disconnected, ["Alpha"], ["isolated"]), 0.0)

    def test_graph_to_records(self) -> None:
        graph = graph_from_triples(self.triples)
        records = graph_to_records(graph)
        self.assertEqual(len(records["nodes"]), graph.number_of_nodes())
        self.assertEqual(len(records["edges"]), graph.number_of_edges())
        self.assertIn("relations", records["edges"][0])


if __name__ == "__main__":
    unittest.main()
