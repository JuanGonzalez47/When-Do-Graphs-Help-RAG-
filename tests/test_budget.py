"""Verify token counting and context packing used by the experiment budgets."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from src.budget import (  # noqa: E402
    budget_metadata,
    count_tokens,
    format_contexts,
    pack_contexts,
    truncate_to_tokens,
)


class BudgetTests(unittest.TestCase):
    """Protect the token-budget constraints applied to model inputs."""

    def test_count_and_truncate(self) -> None:
        text = " ".join(f"word{index}" for index in range(100))
        total = count_tokens(text)
        self.assertGreater(total, 10)
        truncated = truncate_to_tokens(text, 10)
        # Truncation is checked with the experiment tokenizer, not character length.
        self.assertLessEqual(count_tokens(truncated), 10)
        self.assertEqual(truncate_to_tokens(text, total + 10), text)
        self.assertEqual(truncate_to_tokens(text, 0), "")

    def test_pack_respects_budget(self) -> None:
        contexts = [" ".join(["alpha"] * 20), " ".join(["beta"] * 20), " ".join(["gamma"] * 20)]
        selected, used = pack_contexts(contexts, max_tokens=45)
        # The measured count must agree with the sum of tokens actually passed on.
        self.assertLessEqual(used, 45)
        self.assertEqual(used, sum(count_tokens(text) for text in selected))
        self.assertGreaterEqual(len(selected), 1)

    def test_pack_truncates_last(self) -> None:
        contexts = [" ".join(["alpha"] * 20), " ".join(["beta"] * 100)]
        selected, used = pack_contexts(contexts, max_tokens=50, truncate_last=True)
        # Truncating the final context fills remaining budget instead of dropping it.
        self.assertEqual(len(selected), 2)
        self.assertLessEqual(used, 50)
        selected_not, used_not = pack_contexts(contexts, max_tokens=50, truncate_last=False)
        self.assertEqual(len(selected_not), 1)
        self.assertLessEqual(used_not, 50)

    def test_pack_accepts_dicts(self) -> None:
        contexts = [{"text": "alpha " * 10}, {"content": "beta " * 10}]
        selected, _ = pack_contexts(contexts, max_tokens=100)
        self.assertEqual(len(selected), 2)

    def test_format_and_metadata(self) -> None:
        contexts = ["first", "second"]
        formatted = format_contexts(contexts)
        self.assertTrue(formatted.startswith("[1] first"))
        self.assertIn("[2] second", formatted)
        meta = budget_metadata(contexts, max_tokens=100)
        self.assertEqual(meta["contexts"], 2)
        self.assertEqual(meta["budget"], 100)
        self.assertEqual(meta["remaining_tokens"], 100 - meta["context_tokens"])


if __name__ == "__main__":
    unittest.main()
