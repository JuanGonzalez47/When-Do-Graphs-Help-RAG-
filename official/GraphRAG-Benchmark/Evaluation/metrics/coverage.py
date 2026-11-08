import json
from typing import List, Dict
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import Callbacks
from Evaluation.metrics.utils import JSONHandler, binary_array_schema, invoke_json

FACT_EXTRACTION_PROMPT = """
You are given a question and a reference answer. Break down the reference answer into a list of distinct factual statements (facts) that could be independently verified. 
Output them as a JSON list of strings under the 'facts' field.

Example
Input:
  Question: "What causes seasons?"
  Reference: "Seasonal changes result from Earth's axial tilt. This tilt causes different hemispheres to receive varying sunlight."

Output:
{{
  "facts": [
    "Seasonal changes result from Earth's axial tilt",
    "The axial tilt causes different hemispheres to receive varying sunlight"
  ]
}}

### Actual Input
Question: "{question}"
Reference Answer: "{reference}"

### Your Response:
"""

FACT_COVERAGE_PROMPT = """
### Task
For each reference fact, in the exact order given, decide whether the response conveys it.
Return ONLY JSON with a "coverage" array containing exactly one integer per fact:
1 means covered; 0 means not covered. Do not include indices or explanations.

### Actual Input
Question: "{question}"
Response: "{response}"
Reference facts in order: {facts}

### Your Response:
"""

async def compute_coverage_score(
    question: str,
    reference: str,
    response: str,
    llm: BaseLanguageModel,
    callbacks: Callbacks = None,
    max_retries: int = 2
) -> float:
    """
    Calculate coverage score (0.0-1.0) by measuring what percentage of 
    reference facts are covered in the response.
    """
    # Handle edge cases
    if not reference.strip():
        return 1.0  # Perfect coverage for empty reference
    
    # Step 1: Extract facts from reference
    facts = await _extract_facts(
        question, reference, llm, callbacks, max_retries
    )
    
    if not facts:
        raise ValueError("Coverage judge returned no facts for a non-empty reference answer")
    
    # Step 2: Check fact coverage in response
    coverage = await _check_fact_coverage(
        question, facts, response, llm, callbacks, max_retries
    )
    
    # Calculate coverage score
    if coverage:
        attributed = [c["attributed"] for c in coverage]
        return sum(attributed) / len(attributed)
    raise ValueError("Coverage judge returned no valid classifications")

async def _extract_facts(
    question: str,
    reference: str,
    llm: BaseLanguageModel,
    callbacks: Callbacks,
    max_retries: int,
    self_healing: bool = False
) -> List[str]:
    """
    Extract factual statements from the reference answer using an LLM.
    """
    parser = JSONHandler(max_retries=max_retries, self_healing=self_healing)

    prompt = FACT_EXTRACTION_PROMPT.format(
        question=question,
        reference=reference[:3000]  # Avoid overly long prompts
    )

    last_error: Exception | None = None
    for _ in range(max_retries + 1):
        try:
            response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
            parsed = await parser.parse_with_fallbacks(
                response.content,
                llm=llm if self_healing else None,
                callbacks=callbacks
            )
            facts = _validate_facts(
                parsed.get("facts", []) if isinstance(parsed, dict) else parsed
            )
            if not facts:
                raise ValueError("Coverage fact-extraction judge returned no facts")
            return facts
        except Exception as error:
            last_error = error

    raise RuntimeError(
        f"Coverage fact-extraction judge failed after {max_retries + 1} attempts: {last_error}"
    ) from last_error


def _validate_facts(facts: List) -> List[str]:
    """Ensure extracted facts are valid non-empty strings."""
    if not isinstance(facts, list) or not all(
        isinstance(fact, str) and fact.strip() for fact in facts
    ):
        raise ValueError("Coverage fact extraction must return non-empty strings")
    return [fact.strip() for fact in facts]


async def _check_fact_coverage(
    question: str,
    facts: List[str],
    response_text: str,
    llm: BaseLanguageModel,
    callbacks: Callbacks,
    max_retries: int,
    self_healing: bool = False
) -> List[Dict]:
    """
    Check which facts are covered in the given response using an LLM.
    """
    parser = JSONHandler(max_retries=max_retries, self_healing=self_healing)

    prompt = FACT_COVERAGE_PROMPT.format(
        question=question,
        response=response_text[:3000],
        facts=json.dumps(facts)
    )

    last_error: Exception | None = None
    for _ in range(max_retries + 1):
        try:
            response = await invoke_json(
                llm,
                prompt,
                binary_array_schema("coverage", len(facts)),
                callbacks,
            )
            parsed = await parser.parse_with_fallbacks(
                response.content,
                llm=llm if self_healing else None,
                callbacks=callbacks
            )
            ratings = parsed.get("coverage") if isinstance(parsed, dict) else None
            if (
                not isinstance(ratings, list)
                or len(ratings) != len(facts)
                or any(
                    not isinstance(rating, int)
                    or isinstance(rating, bool)
                    or rating not in (0, 1)
                    for rating in ratings
                )
            ):
                raise ValueError("Coverage judge returned an invalid coverage array")
            return [{"attributed": rating} for rating in ratings]
        except Exception as error:
            last_error = error
    raise RuntimeError(
        f"Coverage judge failed after {max_retries + 1} attempts: {last_error}"
    ) from last_error
