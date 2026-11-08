import json
from typing import List, Dict
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import Callbacks
from Evaluation.metrics.utils import (
    JSONHandler,
    binary_array_schema,
    invoke_json,
    string_array_schema,
)

STATEMENT_GENERATOR_PROMPT = """
Given a question and an answer, analyze the complexity of each sentence in the answer. Break down each sentence into one or more fully understandable statements. Ensure that no pronouns are used in any statement. Return ONLY a JSON object with a "statements" array of strings.

Example Input: 
Question: Who was Albert Einstein and what is he best known for?
Answer: He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.

Example Output:
{{"statements":["Albert Einstein was a German-born theoretical physicist.", "Albert Einstein is recognized as one of the greatest and most influential physicists of all time.","Albert Einstein was best known for developing the theory of relativity.","Albert Einstein also made important contributions to the development of the theory of quantum mechanics."]}}

Input Text:
Question:{question}
Answer: {answer}

Return JSON only:
"""

FAITHFULNESS_EXAMPLES = [
    {
        "input": {
            "context": "John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.",
            "statements": [
                "John is majoring in Biology.",
                "John is taking a course on Artificial Intelligence.",
                "John is a dedicated student.",
                "John has a part-time job.",
            ],
        },
        "output": [
            {"statement": "John is majoring in Biology.", "reason": "John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.", "verdict": 0},
            {"statement": "John is taking a course on Artificial Intelligence.", "reason": "The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.", "verdict": 0},
            {"statement": "John is a dedicated student.", "reason": "The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.", "verdict": 1},
            {"statement": "John has a part-time job.", "reason": "There is no information given in the context about John having a part-time job.", "verdict": 0},
        ],
    },
    {
        "input": {
            "context": "Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.",
            "statements": [
                "Albert Einstein was a genius.",
            ],
        },
        "output": [
            {"statement": "Albert Einstein was a genius.", "reason": "The context and statement are unrelated", "verdict": 0},
        ],
    },
]

FAITHFULNESS_EVALUATION_PROMPT = """
Judge whether each statement is supported by the context, preserving the exact order.
Return ONLY JSON with a "verdicts" array containing exactly one integer per statement:
1 means supported; 0 means unsupported. Do not include indices or explanations.

Current Analysis:
Context: {context}
Statements in order: {statements}
"""

async def compute_faithfulness_score(
    question: str,
    answer: str,
    contexts: List[str],
    llm: BaseLanguageModel,
    callbacks: Callbacks = None,
    max_retries: int = 2
) -> float:
    """
    Calculate faithfulness score (0.0-1.0) by measuring what percentage of 
    answer statements are supported by the context.
    """
    # Step 1: Generate atomic statements from answer
    statements = await _generate_statements(
        question, answer, llm, callbacks, max_retries
    )
    
    # Handle edge cases
    if not statements:
        if not answer.strip():
            return 1.0
        raise ValueError("Faithfulness judge returned no statements for a non-empty answer")
    
    context_str = "\n".join(contexts)
    if not context_str.strip():
        return 0.0  # No context means no support
    
    # Step 2: Evaluate statement faithfulness
    verdicts = await _evaluate_statements(
        statements, context_str, llm, callbacks, max_retries
    )
    
    # Calculate faithfulness score
    if verdicts:
        supported = [v["verdict"] for v in verdicts]
        return sum(supported) / len(supported)
    raise ValueError("Faithfulness judge returned no valid verdicts")

async def _generate_statements(
    question: str,
    answer: str,
    llm: BaseLanguageModel,
    callbacks: Callbacks,
    max_retries: int,
    self_healing: bool = False
) -> List[str]:
    """
    Break down the answer into atomic statements using LLM.
    """
    parser = JSONHandler(max_retries=max_retries, self_healing=self_healing)

    prompt = STATEMENT_GENERATOR_PROMPT.format(
        question=question[:500],
        answer=answer[:3000]
    )

    last_error: Exception | None = None
    for _ in range(max_retries + 1):
        try:
            response = await invoke_json(
                llm, prompt, string_array_schema("statements"), callbacks
            )
            parsed = await parser.parse_with_fallbacks(
                response.content,
                llm=llm if self_healing else None,
                callbacks=callbacks
            )
            statements = parsed.get("statements") if isinstance(parsed, dict) else parsed
            if isinstance(statements, list) and all(
                isinstance(statement, str) and statement.strip() for statement in statements
            ) and (statements or not answer.strip()):
                return [statement.strip() for statement in statements]
            raise ValueError("Faithfulness statement judge returned malformed output")
        except Exception as error:
            last_error = error

    raise RuntimeError(
        f"Faithfulness statement-generation judge failed after {max_retries + 1} attempts: {last_error}"
    ) from last_error


async def _evaluate_statements(
    statements: List[str],
    context: str,
    llm: BaseLanguageModel,
    callbacks: Callbacks,
    max_retries: int,
    self_healing: bool = False
) -> List[Dict]:
    """
    Evaluate which statements are supported by the context using LLM.
    """
    parser = JSONHandler(max_retries=max_retries, self_healing=self_healing)

    prompt = FAITHFULNESS_EVALUATION_PROMPT.format(
        context=context[:10000],
        statements=json.dumps(statements)[:5000]
    )

    last_error: Exception | None = None
    for _ in range(max_retries + 1):
        try:
            response = await invoke_json(
                llm,
                prompt,
                binary_array_schema("verdicts", len(statements)),
                callbacks,
            )
            parsed = await parser.parse_with_fallbacks(
                response.content,
                llm=llm if self_healing else None,
                callbacks=callbacks
            )
            raw_verdicts = parsed.get("verdicts") if isinstance(parsed, dict) else None
            validated = _validate_verdicts(raw_verdicts, len(statements))
            return validated
        except Exception as error:
            last_error = error
    raise RuntimeError(
        f"Faithfulness judge failed after {max_retries + 1} attempts: {last_error}"
    ) from last_error


def _validate_verdicts(verdicts: object, statement_count: int) -> List[Dict]:
    """Validate a complete ordered list of binary verdicts."""
    if not isinstance(verdicts, list) or len(verdicts) != statement_count:
        raise ValueError("Faithfulness verdicts must be a list")
    if any(
        not isinstance(verdict, int)
        or isinstance(verdict, bool)
        or verdict not in (0, 1)
        for verdict in verdicts
    ):
        raise ValueError("Faithfulness verdicts must contain only 0 or 1")
    return [{"verdict": int(verdict)} for verdict in verdicts]
