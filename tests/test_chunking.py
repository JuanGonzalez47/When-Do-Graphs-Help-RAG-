"""Verify token-based corpus chunking used to build the retrieval index."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from src.budget import count_tokens  # noqa: E402
from src.chunking import chunk_corpus, chunk_texts, split_text_tokens  # noqa: E402


class ChunkingTests(unittest.TestCase):
    """Protect chunk size, overlap, and corpus-to-chunk conversion."""

    def test_chunk_sizes_and_overlap(self) -> None:
        text = " ".join(f"token{index}" for index in range(2000))
        chunks = split_text_tokens(text, chunk_tokens=128, chunk_overlap=16)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            # Stored offsets and counts must describe the tokenizer's actual chunks.
            self.assertLessEqual(chunk["n_tokens"], 128)
            self.assertEqual(chunk["n_tokens"], count_tokens(chunk["text"]))
            self.assertEqual(chunk["end_token"] - chunk["start_token"], chunk["n_tokens"])
        for previous, current in zip(chunks, chunks[1:]):
            # Preserve the configured overlap so boundaries do not lose context.
            self.assertEqual(previous["end_token"] - current["start_token"], 16)
        self.assertEqual(chunks[-1]["end_token"], count_tokens(text))

    def test_short_text_single_chunk(self) -> None:
        chunks = split_text_tokens("a short passage", chunk_tokens=128, chunk_overlap=16)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["start_token"], 0)

    def test_empty_text(self) -> None:
        self.assertEqual(split_text_tokens("", chunk_tokens=128, chunk_overlap=16), [])

    def test_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            split_text_tokens("text", chunk_tokens=0)
        with self.assertRaises(ValueError):
            split_text_tokens("text", chunk_tokens=8, chunk_overlap=-1)
        with self.assertRaises(ValueError):
            split_text_tokens("text", chunk_tokens=8, chunk_overlap=8)

    def test_chunk_corpus_and_texts(self) -> None:
        corpus = {"a": "alpha beta gamma " * 100, "b": "delta epsilon " * 60}
        chunks = chunk_corpus(corpus, chunk_tokens=32, chunk_overlap=4)
        self.assertEqual(set(chunks), {"a", "b"})
        for source, source_chunks in chunks.items():
            self.assertTrue(source_chunks)
            self.assertEqual(len(chunk_texts(source_chunks)), len(source_chunks))


if __name__ == "__main__":
    unittest.main()
