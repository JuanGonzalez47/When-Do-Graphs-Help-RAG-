import asyncio
import json
import numpy as np
from typing import List
from langchain_core.language_models import BaseLanguageModel

from langchain_core.embeddings import Embeddings
from langchain_core.callbacks import Callbacks
from Evaluation.metrics.utils import JSONHandler, invoke_json, string_array_schema

# F-beta score calculation
def fbeta_score(tp: int, fp: int, fn: int, beta: float = 1.0) -> float:
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-10)

# Statement generation prompt template
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

# Correctness classification prompt template
CORRECTNESS_PROMPT_TEMPLATE = """
Compare each answer statement with the reference statements. A TP is an answer statement supported by at least one reference statement. An FP is unsupported by the reference. An FN is a reference statement not covered by any answer statement.
Return ONLY JSON with exactly two arrays:
{{"answer_supported":[1,0],"reference_covered":[1,0]}}
The answer_supported array must contain exactly one 1 (supported) or 0 (unsupported) for each answer statement, in order. The reference_covered array must contain exactly one 1 (covered) or 0 (not covered) for each reference statement, in order. Do not omit, add, or reorder entries. Do not include explanations.

Current Analysis:
Question: {question}
Answer Statements: {answer}
Reference Statements: {ground_truth}
"""

# Pre-defined examples for correctness classification
CORRECTNESS_EXAMPLES = [
    {
        "input": {
            "question": "What powers the sun and what is its primary function?",
            "answer": [
                "The sun is powered by nuclear fission, similar to nuclear reactors on Earth.",
                "The primary function of the sun is to provide light to the solar system."
            ],
            "ground_truth": [
                "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.",
                "This fusion process in the sun's core releases a tremendous amount of energy.",
                "The energy from the sun provides heat and light, which are essential for life on Earth.",
                "The sun's light plays a critical role in Earth's climate system.",
                "Sunlight helps to drive the weather and ocean currents."
            ]
        },
        "output": {
            "TP": [
                {
                    "statement": "The primary function of the sun is to provide light to the solar system.",
                    "reason": "This statement is somewhat supported by the ground truth mentioning the sun providing light and its roles, though it focuses more broadly on the sun's energy."
                }
            ],
            "FP": [
                {
                    "statement": "The sun is powered by nuclear fission, similar to nuclear reactors on Earth.",
                    "reason": "This statement is incorrect and contradicts the ground truth which states that the sun is powered by nuclear fusion."
                }
            ],
            "FN": [
                {
                    "statement": "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.",
                    "reason": "This accurate description of the sun’s power source is not included in the answer."
                },
                {
                    "statement": "This fusion process in the sun's core releases a tremendous amount of energy.",
                    "reason": "This process and its significance are not mentioned in the answer."
                },
                {
                    "statement": "The energy from the sun provides heat and light, which are essential for life on Earth.",
                    "reason": "The answer only mentions light, omitting the essential aspects of heat and its necessity for life, which the ground truth covers."
                },
                {
                    "statement": "The sun's light plays a critical role in Earth's climate system.",
                    "reason": "This broader impact of the sun’s light on Earth's climate system is not addressed in the answer."
                },
                {
                    "statement": "Sunlight helps to drive the weather and ocean currents.",
                    "reason": "The effect of sunlight on weather patterns and ocean currents is omitted in the answer."
                }
            ]
        }
    },
    {
        "input": {
            "question": "What is the boiling point of water?",
            "answer": [
                "The boiling point of water is 100 degrees Celsius at sea level"
            ],
            "ground_truth": [
                "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
                "The boiling point of water can change with altitude."
            ]
        },
        "output": {
            "TP": [
                {
                    "statement": "The boiling point of water is 100 degrees Celsius at sea level",
                    "reason": "This statement is directly supported by the ground truth which specifies the boiling point of water as 100 degrees Celsius at sea level."
                }
            ],
            "FP": [],
            "FN": [
                {
                    "statement": "The boiling point of water can change with altitude.",
                    "reason": "This additional information about how the boiling point of water can vary with altitude is not mentioned in the answer."
                }
            ]
        }
    }
]

async def compute_answer_correctness(
    question: str,
    answer: str,
    ground_truth: str,
    llm: BaseLanguageModel,
    embeddings: Embeddings,
    weights: List[float] = [0.75, 0.25],
    beta: float = 1.0,
    callbacks: Callbacks = None
) -> float:
    """Compute answer correctness score combining factuality and semantic similarity"""
    if (
        len(weights) != 2
        or any(not np.isfinite(weight) or weight < 0 for weight in weights)
        or sum(weights) <= 0
    ):
        raise ValueError("weights must contain two non-negative values with a positive sum")
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be a positive finite number")
    # Generate statements from answer and ground truth
    answer_statements = await generate_statements(llm, question, answer, callbacks)
    gt_statements = await generate_statements(llm, question, ground_truth, callbacks)

    # Calculate factuality score using statement classification
    factuality_score = await calculate_factuality(
        llm, question, answer_statements, gt_statements, callbacks, beta
    ) if weights[0] != 0 else 0.0
    # Calculate semantic similarity
    similarity_score = await calculate_semantic_similarity(
        embeddings, answer, ground_truth
    ) if weights[1] != 0 else 0.0

    # Combine scores using weighted average
    return float(np.average([factuality_score, similarity_score], weights=weights))

async def generate_statements(
    llm: BaseLanguageModel, question: str, answer:str, callbacks: Callbacks
) -> List[str]:
    """Generate concise factual statements from text"""
    handler = JSONHandler(max_retries=2)
    prompt = STATEMENT_GENERATOR_PROMPT.format(question=question, answer=answer)
    if not answer.strip():
        return []
    last_error: Exception | None = None
    for _ in range(3):
        try:
            response = await invoke_json(
                llm, prompt, string_array_schema("statements"), callbacks
            )
            parsed = await handler.parse_with_fallbacks(response.content)
            statements = parsed.get("statements") if isinstance(parsed, dict) else None
            if (
                isinstance(statements, list)
                and statements
                and all(isinstance(item, str) and item.strip() for item in statements)
            ):
                return [item.strip() for item in statements]
            raise ValueError("Statement judge returned malformed statements")
        except Exception as error:
            last_error = error
    raise RuntimeError(
        f"Answer-correctness statement extraction failed after 3 attempts: {last_error}"
    ) from last_error

async def calculate_factuality(
    llm: BaseLanguageModel,
    question: str,
    answer_stmts: List[str],
    gt_stmts: List[str],
    callbacks: Callbacks,
    beta: float
) -> float:
    """Classify statements and calculate factuality F-beta score"""
    if not answer_stmts and not gt_stmts:
        return 1.0  # Perfect score if both empty

    # Generate classification
    prompt = CORRECTNESS_PROMPT_TEMPLATE.format(
        question=question,
        answer=json.dumps(answer_stmts),
        ground_truth=json.dumps(gt_stmts)
    )
    schema = {
        "type": "object",
        "properties": {
            "answer_supported": {
                "type": "array",
                "items": {"type": "integer", "enum": [0, 1]},
                "minItems": len(answer_stmts),
                "maxItems": len(answer_stmts),
            },
            "reference_covered": {
                "type": "array",
                "items": {"type": "integer", "enum": [0, 1]},
                "minItems": len(gt_stmts),
                "maxItems": len(gt_stmts),
            },
        },
        "required": ["answer_supported", "reference_covered"],
        "additionalProperties": False,
    }
    last_error: Exception | None = None
    for _ in range(3):
        try:
            response = await invoke_json(llm, prompt, schema, callbacks)
            parsed = await JSONHandler().parse_with_fallbacks(response.content)
            if not isinstance(parsed, dict):
                raise ValueError("Answer-correctness judge did not return valid JSON")
            answer_supported = parsed.get("answer_supported")
            reference_covered = parsed.get("reference_covered")
            if not _is_binary_array(answer_supported, len(answer_stmts)):
                raise ValueError("Answer-correctness judge returned an incomplete answer_supported array")
            if not _is_binary_array(reference_covered, len(gt_stmts)):
                raise ValueError("Answer-correctness judge returned an incomplete reference_covered array")
            break
        except Exception as error:
            last_error = error
    else:
        raise RuntimeError(
            f"Answer-correctness classification failed after 3 attempts: {last_error}"
        ) from last_error
    tp = sum(answer_supported)
    fp = len(answer_stmts) - tp
    fn = len(gt_stmts) - sum(reference_covered)
    return fbeta_score(tp, fp, fn, beta)


def _is_binary_array(values: object, expected_count: int) -> bool:
    return (
        isinstance(values, list)
        and len(values) == expected_count
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value in (0, 1)
            for value in values
        )
    )


async def calculate_semantic_similarity(
    embeddings: Embeddings, answer: str, ground_truth: str
) -> float:
    """Compute cosine similarity between answer and ground truth embeddings"""
    a_embed, gt_embed = await asyncio.gather(
        embeddings.aembed_query(answer),
        embeddings.aembed_query(ground_truth)
    )
    cosine_sim = np.dot(a_embed, gt_embed) / (
        np.linalg.norm(a_embed) * np.linalg.norm(gt_embed))
    if not np.isfinite(cosine_sim):
        raise ValueError("Embedding cosine similarity is not finite")
    return float(np.clip((cosine_sim + 1) / 2, 0.0, 1.0))