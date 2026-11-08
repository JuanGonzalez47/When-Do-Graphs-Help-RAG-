"""Verify model-provider configuration, response parsing, and generation caching."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from src.llm import DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL, LLMClient, extract_json  # noqa: E402


class FakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.responses.pop(0)

        class Message:
            pass

        class Choice:
            pass

        class Response:
            pass

        message = Message()
        message.content = content
        choice = Choice()
        choice.message = message
        response = Response()
        response.choices = [choice]
        return response


class FakeChat:
    def __init__(self, responses: list[str]) -> None:
        self.completions = FakeCompletions(responses)


class FakeOpenAI:
    def __init__(self, responses: list[str]) -> None:
        self.chat = FakeChat(responses)


class LLMTests(unittest.TestCase):
    """Protect LLM behavior needed for repeatable local or hosted model runs."""

    def test_extract_json_plain(self) -> None:
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})
        self.assertEqual(extract_json("[1, 2, 3]"), [1, 2, 3])

    def test_extract_json_fences_and_prose(self) -> None:
        text = 'Here is the result:\n```json\n{"a": [1, 2]}\n```\nDone.'
        self.assertEqual(extract_json(text), {"a": [1, 2]})

    def test_extract_json_invalid(self) -> None:
        with self.assertRaises(ValueError):
            extract_json("no json here")

    def test_complete_uses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake = FakeOpenAI(["first response", "second response"])
            client = LLMClient(client=fake, cache_dir=tmp)
            self.assertEqual(client.complete("system", "user"), "first response")
            self.assertEqual(client.complete("system", "user"), "first response")
            # Identical requests must not trigger extra paid/local model generations.
            self.assertEqual(len(fake.chat.completions.calls), 1)
            self.assertEqual(client.complete("system", "other"), "second response")
            self.assertEqual(len(fake.chat.completions.calls), 2)

    def test_complete_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake = FakeOpenAI(['{"answer": 42}'])
            client = LLMClient(client=fake, cache_dir=tmp)
            self.assertEqual(client.complete_json("system", "user"), {"answer": 42})

    def test_missing_api_key_for_openai(self) -> None:
        client = LLMClient(
            client=None,
            cache_dir=tempfile.gettempdir(),
            api_key=None,
            provider="openai",
        )
        client.api_key = None
        with self.assertRaises(RuntimeError):
            client._client()

    def test_ollama_fallback_without_key(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            client = LLMClient(model="gpt-4o-mini", cache_dir=tempfile.gettempdir())
        self.assertEqual(client.provider, "ollama")
        self.assertEqual(client.model, DEFAULT_OLLAMA_MODEL)
        self.assertEqual(client.base_url, DEFAULT_OLLAMA_BASE_URL)
        self.assertEqual(client.api_key, "ollama")

    def test_ollama_model_and_url_from_env(self) -> None:
        env = {
            "OLLAMA_MODEL": "llama3.1:8b",
            "OLLAMA_BASE_URL": "http://127.0.0.1:11434/v1",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            client = LLMClient(cache_dir=tempfile.gettempdir())
        self.assertEqual(client.provider, "ollama")
        self.assertEqual(client.model, "llama3.1:8b")
        self.assertEqual(client.base_url, "http://127.0.0.1:11434/v1")

    def test_explicit_local_model_is_preserved(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            client = LLMClient(model="qwen2.5:3b", cache_dir=tempfile.gettempdir())
        self.assertEqual(client.model, "qwen2.5:3b")

    def test_openai_provider_with_key(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            client = LLMClient(
                model="gpt-4o-mini",
                api_key="test-key",
                cache_dir=tempfile.gettempdir(),
            )
        self.assertEqual(client.provider, "openai")
        self.assertEqual(client.model, "gpt-4o-mini")


if __name__ == "__main__":
    unittest.main()
