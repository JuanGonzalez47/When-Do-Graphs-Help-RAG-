"""Verify the frozen experimental sample and official corpus/question loaders."""

from __future__ import annotations

import hashlib
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from src.data import (  # noqa: E402
    LEVELS,
    SPLITS,
    build_sample,
    load_corpus,
    load_questions,
    load_sample,
    sample_sources,
    validate_sample,
)


class SampleTests(unittest.TestCase):
    """Protect sample size, balance, IDs, evidence fields, and reproducibility."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sample = load_sample()
        cls.records = list(cls.sample["records"])

    def test_total_and_levels(self) -> None:
        self.assertEqual(len(self.records), 400)
        self.assertEqual(self.sample["total"], 400)
        # Each split has equal representation for all four question types.
        report = validate_sample(self.records)
        self.assertEqual(set(report["splits"]), set(SPLITS))
        for split in SPLITS:
            self.assertEqual(report["splits"][split]["total"], 200)
            self.assertEqual(
                report["splits"][split]["by_level"],
                {level: 50 for level in LEVELS},
            )

    def test_ids_unique(self) -> None:
        ids = [record["id"] for record in self.records]
        self.assertEqual(len(ids), len(set(ids)))

    def test_hash_matches_ids(self) -> None:
        ids = self.sample["question_ids"]
        # This digest makes accidental changes to the frozen experimental sample detectable.
        digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        self.assertEqual(digest, self.sample["sha256_ids"])

    def test_sources_cover_corpora(self) -> None:
        sources = sample_sources(self.records)
        self.assertEqual(len(sources["novel"]), 20)
        self.assertEqual(len(sources["medical"]), 1)

    def test_evidence_field_by_split(self) -> None:
        for record in self.records:
            if record["id"].startswith("Novel"):
                self.assertIn("evidence_triple", record)
            else:
                self.assertIn("evidence_relations", record)

    def test_build_sample_reproduces_frozen_ids(self) -> None:
        rebuilt = build_sample(per_level=50, seed=42)
        # Same seed and sampling protocol must reproduce the exact question order.
        self.assertEqual(
            [record["id"] for record in rebuilt["records"]],
            list(self.sample["question_ids"]),
        )


class CorpusTests(unittest.TestCase):
    """Check corpus loading and duplicate-question handling by split."""

    def test_corpus_formats(self) -> None:
        novel = load_corpus("novel")
        medical = load_corpus("medical")
        self.assertEqual(len(novel), 20)
        self.assertEqual(len(medical), 1)
        self.assertTrue(all(isinstance(text, str) and text for text in novel.values()))
        self.assertTrue(all(isinstance(text, str) and text for text in medical.values()))

    def test_question_dedupe(self) -> None:
        for split in SPLITS:
            raw = load_questions(split, dedupe=False)
            deduped = load_questions(split, dedupe=True)
            self.assertLessEqual(len(deduped), len(raw))
            self.assertEqual(len({row["id"] for row in deduped}), len(deduped))
            counts = Counter(row["question_type"] for row in deduped)
            for level in LEVELS:
                self.assertIn(level, counts)


if __name__ == "__main__":
    unittest.main()
