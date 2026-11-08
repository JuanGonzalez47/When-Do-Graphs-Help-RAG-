#!/usr/bin/env python
"""Inspect the configured OpenAI-compatible endpoint and optionally ping the model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from src.llm import LLMClient  


def main() -> None:
    """Display provider/model information and fail clearly if connectivity is broken."""
    parser = argparse.ArgumentParser(description="Check the configured LLM endpoint")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--provider", default=None, choices=["ollama", "openai"])
    parser.add_argument("--ping", action="store_true", help="Send a one-token completion")
    args = parser.parse_args()

    client = LLMClient(
        model=args.model,
        base_url=args.base_url,
        provider=args.provider,
    )
    print(f"provider: {client.provider}")
    print(f"model:    {client.model}")
    print(f"base_url: {client.base_url}")
    try:
        # Listing models detects endpoint/configuration problems before long runs.
        models = client._client().models.list()
        names = [model.id for model in models.data]
        print(f"models:   {', '.join(names[:20]) if names else '(empty list)'}")
        if names and client.model not in names:
            print(f"warning:  model {client.model!r} is not installed")
            print(f"          run: ollama pull {client.model}")
            print(f"          or choose an installed model with OLLAMA_MODEL")
    except Exception as error:
        print(f"models:   unavailable ({error})")
        raise SystemExit(1) from error
    if args.ping:
        try:
            # Disable cache so this verifies a live completion, not a stale response.
            answer = client.complete("Reply with exactly OK.", "OK?", max_tokens=8, use_cache=False)
            print(f"ping:     {answer.strip()[:80]}")
        except Exception as error:
            print(f"ping:     failed ({error})")
            raise SystemExit(1) from error


if __name__ == "__main__":
    main()
