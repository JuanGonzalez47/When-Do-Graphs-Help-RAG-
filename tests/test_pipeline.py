"""Verify experiment execution, output records, resume behavior, and dry runs."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from src.conditions import ExperimentResources  # noqa: E402
from src.graph import graph_from_triples  # noqa: E402
from src.io import read_json, read_jsonl  # noqa: E402
from src.pipeline import OFFICIAL_FIELDS, run_condition, select_per_level  # noqa: E402
from src.retrieval import TfidfRetriever  # noqa: E402

TRIPLES = [
    ("Alpha", "causes", "Beta"),
    ("Beta", "causes", "Gamma"),
    ("Gamma", "inhibits", "Delta"),
]

CHUNKS = [
    {"chunk_id": 0, "source": "Book", "text": "Alpha causes Beta and Beta causes Gamma."},
    {"chunk_id": 1, "source": "Book", "text": "Gamma inhibits Delta."},
]

RECORDS = [
    {
        "id": f"Novel-{index}",
        "source": "Book",
        "question": "What does Alpha cause?",
        "question_type": "Fact Retrieval",
        "answer": "Beta",
        "evidence": "Alpha causes Beta",
    }
    for index in range(5)
]


class FakeClient:
    """Return a fixed answer so pipeline behavior is tested without model calls."""

    model = "fake-model"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system: str, user: str, **kwargs) -> str:
        self.calls += 1
        return "Beta"


class PipelineTests(unittest.TestCase):
    """Protect the path from selected records to persisted experiment predictions."""

    def setUp(self) -> None:
        # Temporary outputs prevent a test run from overwriting research artifacts.
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # Small deterministic corpus/graph exercise the same retrieval resources as runs.
        self.resources = ExperimentResources(
            retriever=TfidfRetriever().fit(CHUNKS),
            graphs={"Book": graph_from_triples(TRIPLES)},
            rag_top_k=2,
            graph_hops=2,
            seed_k=2,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_cell(self, condition: str = "rag", dry_run: bool = False, resume: bool = False):
        # Route every generated file to the temporary directory created for this test.
        return run_condition(
            RECORDS,
            condition,
            200,
            self.resources,
            "novel",
            client=None if dry_run else FakeClient(),
            output_dir=self.root / "outputs",
            raw_dir=self.root / "raw",
            manifest_path=self.root / "manifest.csv",
            dry_run=dry_run,
            resume=resume,
        )

    def test_rag_smoke_five_questions(self) -> None:
        summary = self.run_cell("rag")
        self.assertEqual(summary["records"], 5)
        self.assertEqual(summary["errors"], 0)
        output = read_json(summary["path"])
        self.assertEqual(len(output), 5)
        # The persisted schema must remain consumable by the official evaluator.
        for prediction in output:
            for field in OFFICIAL_FIELDS:
                self.assertIn(field, prediction)
            self.assertEqual(prediction["generated_answer"], "Beta")
            self.assertEqual(prediction["ground_truth"], "Beta")
            self.assertIsInstance(prediction["context"], list)
        raw = read_jsonl(self.root / "raw" / "novel" / "rag_200.jsonl")
        # Raw predictions and manifest enable recovery and traceability of long runs.
        self.assertEqual(len(raw), 5)
        self.assertTrue((self.root / "manifest.csv").exists())

    def test_dry_run(self) -> None:
        summary = self.run_cell("graph_full", dry_run=True)
        output = read_json(summary["path"])
        self.assertEqual(len(output), 5)
        self.assertTrue(all(prediction["generated_answer"] for prediction in output))

    def test_resume_does_not_duplicate(self) -> None:
        self.run_cell("rag")
        summary = self.run_cell("rag", resume=True)
        self.assertEqual(summary["records"], 5)
        raw = read_jsonl(self.root / "raw" / "novel" / "rag_200.jsonl")
        # Resuming a completed condition must not append duplicate question IDs.
        self.assertEqual(len(raw), 5)

    def test_rerun_replaces_raw(self) -> None:
        self.run_cell("rag")
        summary = self.run_cell("rag")
        raw = read_jsonl(self.root / "raw" / "novel" / "rag_200.jsonl")
        # A fresh run replaces prior raw output rather than silently mixing runs.
        self.assertEqual(len(raw), 5)
        self.assertEqual(summary["records"], 5)

    def test_graph_condition_smoke(self) -> None:
        summary = self.run_cell("graph_adaptive")
        self.assertEqual(summary["records"], 5)
        output = read_json(summary["path"])
        self.assertTrue(all(prediction["condition"] == "graph_adaptive" for prediction in output))

    def test_select_per_level(self) -> None:
        records = [
            {"id": f"{level}-{index}", "question_type": level}
            for level in ["Fact Retrieval", "Complex Reasoning", "Contextual Summarize", "Creative Generation"]
            for index in range(3)
        ]
        selected = select_per_level(records, 2)
        # Verify the requested per-type sample size and deterministic level ordering.
        self.assertEqual(len(selected), 8)
        self.assertEqual([record["id"] for record in selected[:3]], ["Fact Retrieval-0", "Fact Retrieval-1", "Complex Reasoning-0"])


if __name__ == "__main__":
    unittest.main()
