"""Verify answer generation and the prediction records consumed by evaluation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from src.generation import build_prediction, generate_answer, prepare_contexts  # noqa: E402


class FakeClient:
    """Replace the external LLM so tests stay deterministic and cost-free."""

    def __init__(self, answer: str = "generated") -> None:
        self.answer = answer
        self.calls = []

    def complete(self, system: str, user: str, **kwargs) -> str:
        self.calls.append((system, user, kwargs))
        return self.answer


class GenerationTests(unittest.TestCase):
    """Protect prompt context selection, budget preparation, and output schema."""

    def test_generate_answer(self) -> None:
        client = FakeClient("Paris")
        answer = generate_answer("What is the capital of France?", ["France's capital is Paris."], client)
        self.assertEqual(answer, "Paris")
        self.assertIn("What is the capital of France?", client.calls[0][1])
        self.assertIn("[1]", client.calls[0][1])

    def test_generate_answer_uses_prompt_contexts(self) -> None:
        client = FakeClient("ok")
        generate_answer("q", ["full context"], client, prompt_contexts=["trimmed context"])
        # Prompt-only contexts may be budget-trimmed; the original is not sent again.
        self.assertIn("trimmed context", client.calls[0][1])
        self.assertNotIn("full context", client.calls[0][1])

    def test_prepare_contexts(self) -> None:
        contexts, used = prepare_contexts(["alpha " * 50, "beta " * 50], budget=20)
        self.assertLessEqual(used, 20)
        self.assertTrue(contexts)

    def test_build_prediction(self) -> None:
        record = {
            "id": "Novel-1",
            "source": "Novel-1",
            "question": "Who?",
            "question_type": "Fact Retrieval",
            "answer": "Alice",
            "evidence": "Alice did it.",
        }
        prediction = build_prediction(record, ["context"], "Alice", "rag", 1024)
        # Evaluator input keeps a standardized field and avoids an extra answer alias.
        self.assertEqual(prediction["generated_answer"], "Alice")
        self.assertEqual(prediction["ground_truth"], "Alice")
        self.assertEqual(prediction["context"], ["context"])
        self.assertEqual(prediction["condition"], "rag")
        self.assertNotIn("answer", prediction)

    def test_build_prediction_evidence_fallback(self) -> None:
        record = {
            "id": "Medical-1",
            "question": "q",
            "question_type": "Fact Retrieval",
            "answer": "a",
            "evidence_relations": "relation",
        }
        prediction = build_prediction(record, [], "a", "rag", 1024)
        self.assertEqual(prediction["evidence"], "relation")


if __name__ == "__main__":
    unittest.main()
