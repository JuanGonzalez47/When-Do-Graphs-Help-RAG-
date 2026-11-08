from typing import List, Dict

from langchain_core.callbacks import Callbacks
from langchain_core.language_models import BaseLanguageModel

from Evaluation.metrics.utils import JSONHandler, binary_array_schema, invoke_json

EVIDENCE_RECALL_PROMPT = """
### Task
You are given a list of evidences and a Context. For each evidence, determine whether it can be attributed to the Context, preserving the exact order.

Respond ONLY with JSON containing an "attributed" array with exactly one integer per evidence. Use 1 if supported by the Context and 0 otherwise. Do not include indices or explanations.

### Example
Input:
Context: "Einstein won the Nobel Prize in 1921 for physics."
Evidence: ["Einstein received the Nobel Prize", "He was born in Germany"]

Output:
{{
"attributed": [1, 0]
}}

### Actual Input
Context: "{context}"

Evidence in order: {evidence}

Question: "{question}" (for reference only)

### Your Response:
"""

async def compute_evidence_recall(
    question: str,
    contexts: List[str],
    reference_evidence: List[str],
    llm: BaseLanguageModel,
    callbacks: Callbacks = None,
    max_retries: int = 2
) -> float:
    """
    Calculate context recall score (0.0-1.0) by measuring what percentage of 
    reference evidence are supported by the context.
    """
    # Handle edge cases
    if not all(isinstance(evidence, str) and evidence.strip() for evidence in reference_evidence):
        raise ValueError("reference_evidence must contain non-empty strings")
    if not reference_evidence:
        raise ValueError("Evidence recall is undefined when the reference evidence list is empty")
    if isinstance(contexts, list):
        if not all(isinstance(context, str) for context in contexts):
            raise ValueError("contexts must contain only strings")
        context_str = "\n".join(contexts)
    elif isinstance(contexts, str):
        context_str = contexts
    else:
        raise ValueError("contexts must be a list of strings or a single string.")

    # No context means no attribution
    if not context_str.strip():
        return 0.0
    prompt = EVIDENCE_RECALL_PROMPT.format(
        question=question,
        context=context_str[:20000],  # Truncate long contexts # TODO : smarter truncation
        evidence=reference_evidence,  # Truncate long answers TODO how?
    )

    # Get LLM classification with retries
    classifications = await _get_classifications(
        prompt, len(reference_evidence), llm, callbacks, max_retries
    )

    # Calculate recall score
    if classifications:
        attributed = [c["attributed"] for c in classifications]
        return sum(attributed) / len(attributed)
    raise ValueError("Evidence recall judge returned no valid classifications")

async def _get_classifications(
    prompt: str,
    evidence_count: int,
    llm: BaseLanguageModel,
    callbacks: Callbacks,
    max_retries: int,
    self_healing: bool = False
) -> List[Dict]:
    """
    Get valid classifications from LLM with retries using RobustJSONHandler.
    """
    parser = JSONHandler(max_retries=max_retries, self_healing=self_healing)

    last_error: Exception | None = None
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    for _ in range(max_retries + 1):
        try:
            response = await invoke_json(
                llm,
                prompt,
                binary_array_schema("attributed", evidence_count),
                callbacks,
            )
            parsed = await parser.parse_with_fallbacks(
                response.content,
                llm=llm if self_healing else None,
                callbacks=callbacks
            )
            attributions = parsed.get("attributed") if isinstance(parsed, dict) else None
            return _validate_classifications(attributions, evidence_count)
        except Exception as error:
            last_error = error
    raise RuntimeError(
        f"Evidence recall judge failed after {max_retries + 1} attempts: {last_error}"
    ) from last_error

def _validate_classifications(classifications: List, evidence_count: int) -> List[Dict]:
    """
    Ensure classifications have required fields and proper types.
    """
    if not isinstance(classifications, list) or len(classifications) != evidence_count:
        raise ValueError("Evidence recall must return one attribution per evidence")
    if any(
        not isinstance(attributed, int)
        or isinstance(attributed, bool)
        or attributed not in (0, 1)
        for attributed in classifications
    ):
        raise ValueError("Evidence recall attributions must contain only 0 or 1")
    return [{"attributed": int(attributed)} for attributed in classifications]