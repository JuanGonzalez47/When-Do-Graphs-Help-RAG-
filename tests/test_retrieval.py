"""Verify TF-IDF chunk retrieval used by the text-RAG baseline."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from src.retrieval import TfidfRetriever, retrieved_to_contexts  # noqa: E402


class RetrievalTests(unittest.TestCase):
    """Check ranking, source filtering, and conversion of hits to contexts."""

    def setUp(self) -> None:
        # Two sources and overlapping terms make ranking and source filtering observable.
        self.chunks = [
            {"chunk_id": 0, "source": "a", "text": "the cat sat on the mat"},
            {"chunk_id": 1, "source": "a", "text": "dogs bark loudly at night"},
            {"chunk_id": 2, "source": "b", "text": "the cat chased a mouse"},
        ]
        self.retriever = TfidfRetriever().fit(self.chunks)

    def test_search_returns_ranked_results(self) -> None:
        results = self.retriever.search("cat mat", top_k=2)
        self.assertEqual(len(results), 2)
        # The exact-match chunk should rank first, with scores in descending order.
        self.assertEqual(results[0].chunk_id, 0)
        self.assertGreaterEqual(results[0].score, results[1].score)

    def test_search_source_filter(self) -> None:
        results = self.retriever.search("cat", top_k=5, source="b")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source, "b")

    def test_search_min_score(self) -> None:
        results = self.retriever.search("cat", top_k=5, min_score=0.5)
        # A retrieval threshold must exclude every result below the cutoff.
        self.assertTrue(all(result.score >= 0.5 for result in results))

    def test_metadata_preserved(self) -> None:
        results = self.retriever.search("dogs", top_k=1)
        self.assertEqual(results[0].metadata, {})

    def test_empty_fit_raises(self) -> None:
        with self.assertRaises(ValueError):
            TfidfRetriever().fit([])

    def test_search_before_fit_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            TfidfRetriever().search("cat")

    def test_retrieved_to_contexts(self) -> None:
        results = self.retriever.search("cat", top_k=2)
        contexts = retrieved_to_contexts(results)
        self.assertEqual(len(contexts), 2)
        self.assertTrue(all(isinstance(context, str) for context in contexts))


if __name__ == "__main__":
    unittest.main()
