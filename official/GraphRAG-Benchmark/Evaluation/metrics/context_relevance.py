from typing import List, Optional, Union

import numpy as np
from langchain_core.callbacks import Callbacks
from langchain_core.language_models import BaseLanguageModel

from Evaluation.metrics.utils import JSONHandler

CONTEXT_RELEVANCE_PROMPT = """
### Instructions
You are a world class expert designed to evaluate the relevance score of a Context in order to answer the Question.
Your task is to determine if the Context contains proper information to answer the Question.
Do not rely on your previous knowledge about the Question.
Use only what is written in the Context and in the Question.

Scoring rules:
0. If the context does not contain any relevant information to answer the question, score 0.
1. If the context partially contains relevant information to answer the question, score 1.
2. If the context fully contains relevant information to answer the question, score 2.

Output format:
You must output strictly in JSON format with a single key "score".
No explanation, no additional text.

Example:
Question: What is the capital of France?
Context: Paris is the capital of France.
Output:
{{ "score": 2 }}

Now evaluate the following:
Question: {question}
Context: {context}
/no_think
"""

 
async def compute_context_relevance(
    question: str,
    contexts: List[str],
    llm: BaseLanguageModel,
    callbacks: Callbacks = None,
    max_retries: int = 2
) -> float:
    """
    Evaluate the relevance of retrieved contexts for answering a question.
    Returns a score between 0.0 (irrelevant) and 1.0 (fully relevant).
    """
    if isinstance(contexts, list):
        if not all(isinstance(context, str) for context in contexts):
            raise ValueError("contexts must contain only strings")
        context_str = "\n".join(contexts)
    elif isinstance(contexts, str):
        context_str = contexts
    else:
        raise ValueError("contexts must be a list of strings or a single string.")
    if not question.strip() or not context_str.strip():
        return 0.0
    
    prompt = CONTEXT_RELEVANCE_PROMPT.format(
        question=question,
        context=context_str[:20000], # Truncate long contexts # TODO : smarter truncation
    )

    # Get two independent relevance ratings from LLM
    rating1 = await _get_llm_rating(prompt, llm, callbacks, max_retries)
    rating2 = await _get_llm_rating(prompt, llm, callbacks, max_retries)

    # Process ratings (0-2 scale) and convert to 0-1 scale
    scores = [rating1, rating2]
    if any(score is None for score in scores):
        raise RuntimeError("Context relevance judge did not return two valid ratings")
    return sum(score / 2 for score in scores if score is not None) / len(scores)


async def _get_llm_rating(
    prompt: str,
    llm: BaseLanguageModel,
    callbacks: Callbacks,
    max_retries: int,
    self_healing: bool = False
) -> float:
    """
    Get a single relevance rating from LLM with retries.
    """
    parser = JSONHandler(max_retries=max_retries, self_healing=self_healing)

    last_error: Exception | None = None
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    for _ in range(max_retries + 1):
        try:
            response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
            parsed = await parser.parse_with_fallbacks(
                response.content,
                llm=llm if self_healing else None,
                callbacks=callbacks
            )
            rating = _normalize_rating(parsed)
            if rating is None:
                raise ValueError("Context relevance judge returned an invalid score")
            return rating
        except Exception as error:
            last_error = error
    raise RuntimeError(
        f"Context relevance judge failed after {max_retries + 1} attempts: {last_error}"
    ) from last_error


def _normalize_rating(parsed: Union[dict, list, str, None]) -> Optional[float]:
    """
    Normalize parsed content to extract a valid rating (0-2).
    """
    # Case 1: JSON dict with "rating" or "score"
    if isinstance(parsed, dict):
        score = parsed.get("rating", parsed.get("score"))
        if _is_valid_rating(score):
            return float(score)

    return None

def _is_valid_rating(value) -> bool:
    """Check if value is one of the defined scores: 0, 1, or 2."""
    if isinstance(value, bool):
        return False
    try:
        ivalue = float(value)
        return ivalue in (0.0, 1.0, 2.0)
    except (TypeError, ValueError):
        return False