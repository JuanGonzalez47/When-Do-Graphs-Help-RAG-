import asyncio
import argparse
import json
import math
import numpy as np
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from Evaluation.metrics import compute_answer_correctness, compute_coverage_score, compute_faithfulness_score, compute_rouge_score
from langchain_ollama import OllamaEmbeddings
from Evaluation.llm import OllamaClient, OllamaWrapper

SEED = 42

async def evaluate_dataset(
    dataset: Dataset,
    metrics: List[str],
    llm: Any,
    embeddings: Embeddings,
    max_concurrent: int = 1,
    detailed_output: bool = False
) -> Dict[str, Any]:
    """Evaluate samples while preserving metric-level failures in the report."""
    results: dict[str, list[float]] = {metric: [] for metric in metrics}
    detailed_results: list[dict[str, Any]] = []
    failed_samples: list[dict[str, Any]] = []
    
    ids = dataset["id"]
    questions = dataset["question"]
    answers = dataset["answer"]
    contexts_list = dataset["contexts"]
    ground_truths = dataset["ground_truth"]
    
    total_samples = len(questions)
    print(f"\nStarting evaluation of {total_samples} samples...")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def evaluate_with_semaphore(i):
        async with semaphore:
            return await evaluate_sample(
                question=questions[i],
                answer=answers[i],
                contexts=contexts_list[i],
                ground_truth=ground_truths[i],
                metrics=metrics,
                llm=llm,
                embeddings=embeddings
            )

    async def evaluate_index(i: int) -> tuple[int, dict[str, Any] | None, str | None]:
        try:
            return i, await evaluate_with_semaphore(i), None
        except Exception as error:
            return i, None, f"{type(error).__name__}: {error}"

    tasks = [evaluate_index(i) for i in range(total_samples)]
    completed = 0

    for future in asyncio.as_completed(tasks):
        index, result, fatal_error = await future
        completed += 1
        print(f"✅ Completed sample {completed}/{total_samples} - {(completed/total_samples)*100:.1f}%")
        if fatal_error:
            failed_samples.append({"id": ids[index], "errors": {"sample": fatal_error}})
            continue
        assert result is not None
        sample_metrics = result["metrics"]
        sample_errors = result["errors"]
        for metric, score in sample_metrics.items():
            if score is not None:
                results[metric].append(score)
        if sample_errors:
            failed_samples.append({"id": ids[index], "errors": sample_errors})
        if detailed_output:
            detailed_results.append(
                {
                    "id": ids[index],
                    "question": questions[index],
                    "ground_truth": ground_truths[index],
                    "generated_answer": answers[index],
                    "contexts": contexts_list[index],
                    "metrics": sample_metrics,
                    "errors": sample_errors,
                }
            )

    averages = {
        metric: (float(np.mean(scores)) if scores else None)
        for metric, scores in results.items()
    }
    return {
        "average_scores": averages,
        "metric_counts": {
            metric: {"valid": len(scores), "failed": total_samples - len(scores)}
            for metric, scores in results.items()
        },
        "total_samples": total_samples,
        "failed_samples": failed_samples,
        **({"detailed": detailed_results} if detailed_output else {}),
    }

async def evaluate_sample(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
    metrics: List[str],
    llm: Any,
    embeddings: Embeddings
) -> Dict[str, Any]:
    """Evaluate the metric scores for a single sample."""
    results = {}
    
    tasks = {}
    if "rouge_score" in metrics:
        tasks["rouge_score"] = compute_rouge_score(answer, ground_truth)
    
    if "answer_correctness" in metrics:
        tasks["answer_correctness"] = compute_answer_correctness(
            question, answer, ground_truth, llm, embeddings
        )
    
    if "coverage_score" in metrics:
        tasks["coverage_score"] = compute_coverage_score(
            question, ground_truth, answer, llm
        )
    
    if "faithfulness" in metrics:
        tasks["faithfulness"] = compute_faithfulness_score(
            question, answer, contexts, llm
        )
    
    task_results = []
    for task in tasks.values():
        try:
            task_results.append(await task)
        except Exception as error:
            task_results.append(error)
    errors: dict[str, str] = {}
    for metric, value in zip(tasks, task_results):
        if isinstance(value, BaseException):
            errors[metric] = f"{type(value).__name__}: {value}"
        elif (
            not isinstance(value, (int, float, np.number))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            errors[metric] = f"Invalid metric score: {value!r}"
        else:
            results[metric] = float(value)

    for metric in metrics:
        results.setdefault(metric, None)
    return {"metrics": results, "errors": errors}

async def main(args: argparse.Namespace):
    """Main evaluation function that accepts command-line arguments."""
    if args.num_samples is not None and args.num_samples <= 0:
        raise ValueError("--num_samples must be a positive integer")
    metric_config = {
        'Fact Retrieval': ["rouge_score", "answer_correctness"],
        'Complex Reasoning': ["rouge_score", "answer_correctness"],
        'Contextual Summarize': ["answer_correctness", "coverage_score"],
        'Creative Generation': ["answer_correctness", "coverage_score", "faithfulness"]
    }
    print(f"Loading evaluation data from {args.data_file}...")
    with open(args.data_file, encoding="utf-8") as handle:
        file_data = json.load(handle)
    if not isinstance(file_data, list) or not file_data:
        raise ValueError("Evaluation input must be a non-empty JSON list")
    required = ("id", "question", "question_type", "generated_answer", "ground_truth", "context")
    seen_ids: set[str] = set()
    grouped_data: dict[str, list[dict[str, Any]]] = {}
    for index, item in enumerate(file_data):
        if not isinstance(item, dict):
            raise ValueError(f"Input row {index} must be a JSON object")
        missing = [field for field in required if field not in item]
        if missing:
            raise ValueError(f"Input row {index} is missing required fields: {missing}")
        if not all(isinstance(item[field], str) for field in required[:-1]):
            raise ValueError(f"Input row {index} has non-string required fields")
        if not item["id"].strip() or not item["question"].strip() or not item["question_type"].strip():
            raise ValueError(f"Input row {index} has a blank id, question, or question_type")
        if not isinstance(item["context"], list) or not all(
            isinstance(context, str) for context in item["context"]
        ):
            raise ValueError(f"Input row {index} context must be a list of strings")
        record_id = str(item["id"])
        if record_id in seen_ids:
            raise ValueError(f"Duplicate question id in evaluation input: {record_id}")
        seen_ids.add(record_id)
        grouped_data.setdefault(item["question_type"], []).append(item)
    unknown_types = set(grouped_data) - set(metric_config)
    if unknown_types:
        raise ValueError(f"Unsupported question types in input: {sorted(unknown_types)}")
    if set(grouped_data) != set(metric_config):
        raise ValueError("Evaluation input must contain all four benchmark question types")

    if args.mode == "API":
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError("LLM_API_KEY environment variable is not set")
        from pydantic import SecretStr
        llm = ChatOpenAI(
            model=args.model,
            base_url=args.base_url,
            api_key=SecretStr(api_key),
            temperature=0.0,
            max_retries=3,
            timeout=30,
            model_kwargs={"top_p": 1, "seed": SEED, "presence_penalty": 0, "frequency_penalty": 0},
        )
        embedding = HuggingFaceBgeEmbeddings(model_name=args.embedding_model)
    else:
        ollama_client = OllamaClient(base_url=args.base_url)
        llm = OllamaWrapper(
            ollama_client,
            args.model,
            default_options={
                "temperature": 0.0,
                "top_p": 1,
                "num_ctx": 32768,
                "seed": SEED,
                "format": "json",
            },
        )
        embedding = OllamaEmbeddings(model=args.embedding_model, base_url=args.base_url)

    all_results: dict[str, Any] = {}
    any_failures = False
    sampler = random.Random(SEED)
    try:
        for question_type, group_items in grouped_data.items():
            print(f"\n{'='*50}\nEvaluating question type: {question_type}\n{'='*50}")
            if args.num_samples is not None:
                if args.num_samples > len(group_items):
                    raise ValueError(
                        f"--num_samples={args.num_samples} exceeds "
                        f"{len(group_items)} records for {question_type}"
                    )
                group_items = sampler.sample(group_items, args.num_samples)
            if not group_items:
                raise ValueError(f"No evaluation records for question type {question_type}")
            data = {
                "id": [item["id"] for item in group_items],
                "question": [item["question"] for item in group_items],
                "answer": [item["generated_answer"] for item in group_items],
                "contexts": [item["context"] for item in group_items],
                "ground_truth": [item["ground_truth"] for item in group_items],
            }
            results = await evaluate_dataset(
                dataset=Dataset.from_dict(data),
                metrics=metric_config[question_type],
                llm=llm,
                embeddings=embedding,
                detailed_output=args.detailed_output,
            )
            all_results[question_type] = results
            any_failures = any_failures or bool(results["failed_samples"])
            print("\nAverage scores (valid count / total):")
            for metric, score in results["average_scores"].items():
                counts = results["metric_counts"][metric]
                rendered = f"{score:.4f}" if score is not None else "FAILED"
                print(f"  {metric}: {rendered} ({counts['valid']}/{results['total_samples']})")

        report = {
            "metadata": {
                "mode": args.mode,
                "model": args.model,
                "embedding_model": args.embedding_model,
                "input_file": str(args.data_file),
                "num_samples_per_question_type": args.num_samples,
                "sample_seed": SEED,
                "complete": not any_failures,
            },
            "question_types": all_results,
        }
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        print(f"\nSaved evaluation report: {output_path}")
    finally:
        if args.mode == "ollama":
            await llm.close()
    if any_failures:
        raise SystemExit("Evaluation finished with failed metrics; inspect failed_samples in the report")
    print("\nEvaluation complete with no failed metrics.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate RAG performance using various metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--mode", 
        required=True,
        choices=["API", "ollama"],
        type=str,
        default="API",
        help="Use API or ollama for LLM"
    )

    parser.add_argument(
        "--model", 
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use for evaluation"
    )
    
    parser.add_argument(
        "--base_url", 
        type=str,
        default="https://api.openai.com/v1",
        help="Base URL for the OpenAI API"
    )
    
    parser.add_argument(
        "--embedding_model", 
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="HuggingFace model for embeddings"
    )
    
    parser.add_argument(
        "--data_file", 
        type=str,
        required=True,
        help="Path to JSON file containing evaluation data"
    )
    
    parser.add_argument(
        "--output_file", 
        type=str,
        default="evaluation_results.json",
        help="Path to save evaluation results"
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to use for evaluation"
    )

    parser.add_argument(
        "--detailed_output",
        action="store_true",
        help="Whether to include detailed output"
    )
    
    args = parser.parse_args()
    
    asyncio.run(main(args))
