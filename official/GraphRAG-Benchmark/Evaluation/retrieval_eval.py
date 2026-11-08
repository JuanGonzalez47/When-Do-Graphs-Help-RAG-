import asyncio
import argparse
import json
import math
import numpy as np
import os
import random
from pathlib import Path
from typing import Dict, List, Any
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from Evaluation.metrics import compute_context_relevance, compute_evidence_recall
from langchain_ollama import OllamaEmbeddings
from Evaluation.llm import OllamaClient,OllamaWrapper

SEED = 42

async def evaluate_dataset(
    dataset: Dataset,
    llm: Any,
    embeddings: Embeddings,
    max_concurrent: int = 1,
    detailed_output: bool = False
) -> Dict[str, Any]:
    """Evaluate retrieval metrics and report every invalid or failed judgment."""
    results: dict[str, list[float]] = {
        "context_relevancy": [],
        "evidence_recall": [],
    }
    detailed_results: list[dict[str, Any]] = []
    failed_samples: list[dict[str, Any]] = []
    
    ids = dataset["id"]
    questions = dataset["question"]
    contexts_list = dataset["contexts"]
    evidences = dataset["evidences"]

    total_samples = len(questions)
    print(f"\nStarting evaluation of {total_samples} samples...")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def evaluate_with_semaphore(i):
        async with semaphore:
            return await evaluate_sample(
                question=questions[i],
                contexts=contexts_list[i],
                evidences=evidences[i],
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
            detailed_results.append({
                "id": ids[index],
                "question": questions[index],
                "contexts": contexts_list[index],
                "evidences": evidences[index],
                "metrics": sample_metrics,
                "errors": sample_errors,
            })

    return {
        "average_scores": {
            metric: (float(np.mean(scores)) if scores else None)
            for metric, scores in results.items()
        },
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
    contexts: List[str],
    evidences: List[str],
    llm: Any,
    embeddings: Embeddings
) -> Dict[str, Any]:
    """Evaluate retrieval metrics for a single sample"""
    # Evaluate both metrics in parallel
    relevance_task = compute_context_relevance(question, contexts, llm)
    recall_task = compute_evidence_recall(question, contexts, evidences, llm)
    
    # Wait for both tasks to complete
    values = await asyncio.gather(
        relevance_task,
        recall_task,
        return_exceptions=True,
    )
    metrics: dict[str, float | None] = {}
    errors: dict[str, str] = {}
    for metric, value in zip(("context_relevancy", "evidence_recall"), values):
        if isinstance(value, BaseException):
            metrics[metric] = None
            errors[metric] = f"{type(value).__name__}: {value}"
        elif (
            not isinstance(value, (int, float, np.number))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            metrics[metric] = None
            errors[metric] = f"Invalid metric score: {value!r}"
        else:
            metrics[metric] = float(value)
    print(f"Metrics: {metrics}; errors: {errors}")
    return {"metrics": metrics, "errors": errors}

async def main(args: argparse.Namespace):
    """Main retrieval evaluation function"""
    if args.num_samples is not None and args.num_samples <= 0:
        raise ValueError("--num_samples must be a positive integer")
    if args.mode == "API":
        # Check API key
        if not os.getenv("LLM_API_KEY"):
            raise ValueError("LLM_API_KEY environment variable is not set")
        
        # Initialize models
        # Wrap API key in SecretStr to satisfy type hints
        from pydantic import SecretStr
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError("LLM_API_KEY environment variable is not set")
        llm = ChatOpenAI(
            model=args.model,
            base_url=args.base_url,
            api_key=SecretStr(api_key),
            temperature=0.0,
            max_retries=3,
            timeout=30,
            model_kwargs={
                "top_p": 1,
                "seed": SEED,
                "presence_penalty": 0,
                "frequency_penalty": 0
            }
        )
        
        # Initialize the embedding model
        embedding = HuggingFaceBgeEmbeddings(model_name=args.embedding_model)

    elif args.mode == "ollama":
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
            }
        )
        ollama_embeddings = OllamaEmbeddings(
            model=args.embedding_model,
            base_url=args.base_url
        )
        embedding = LangchainEmbeddingsWrapper(embeddings=ollama_embeddings)
        
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    # Load evaluation data
    print(f"Loading evaluation data from {args.data_file}...")
    with open(args.data_file, encoding="utf-8") as f:
        file_data = json.load(f)
    if not isinstance(file_data, list) or not file_data:
        raise ValueError("Evaluation input must be a non-empty JSON list")
    seen_ids: set[str] = set()
    required = ("id", "question", "question_type", "context", "evidence")
    for index, item in enumerate(file_data):
        if not isinstance(item, dict):
            raise ValueError(f"Input row {index} must be a JSON object")
        missing = [field for field in required if field not in item]
        if missing:
            raise ValueError(f"Input row {index} is missing required fields: {missing}")
        if not all(isinstance(item[field], str) for field in ("id", "question", "question_type")):
            raise ValueError(f"Input row {index} has non-string id, question, or question_type")
        if not item["id"].strip() or not item["question"].strip() or not item["question_type"].strip():
            raise ValueError(f"Input row {index} has a blank id, question, or question_type")
        if not isinstance(item["context"], list) or not all(
            isinstance(context, str) for context in item["context"]
        ):
            raise ValueError(f"Input row {index} context must be a list of strings")
        if not isinstance(item["evidence"], list) or not all(
            isinstance(evidence, str) and evidence.strip() for evidence in item["evidence"]
        ):
            raise ValueError(f"Input row {index} evidence must be a non-empty list of strings")
        if item["id"] in seen_ids:
            raise ValueError(f"Duplicate question id in evaluation input: {item['id']}")
        seen_ids.add(item["id"])
    
    # Group data by question type
    grouped_data = {}
    for item in file_data:
        q_type = item.get("question_type", "Uncategorized")
        if q_type not in grouped_data:
            grouped_data[q_type] = []
        grouped_data[q_type].append(item)
    expected_types = {
        "Fact Retrieval",
        "Complex Reasoning",
        "Contextual Summarize",
        "Creative Generation",
    }
    if set(grouped_data) != expected_types:
        raise ValueError(
            "Evaluation input must contain all four question types; "
            f"found {sorted(grouped_data)}"
        )
    
    all_results = {}
    sampler = random.Random(SEED)
    
    # Evaluate each question type
    for question_type in list(grouped_data.keys()):
        print(f"\n{'='*50}")
        print(f"Evaluating question type: {question_type}")
        print(f"{'='*50}")
        
        # Prepare data from grouped items
        group_items = grouped_data[question_type]
        if args.num_samples is not None and args.num_samples > len(group_items):
            raise ValueError(
                f"--num_samples={args.num_samples} exceeds {len(group_items)} "
                f"records for {question_type}"
            )

        ids = [item['id'] for item in group_items]
        questions = [item['question'] for item in group_items]
        evidences = [item['evidence'] for item in group_items]
        contexts = [item['context'] for item in group_items]
        
        # Create dataset
        data = {
            "id": ids,
            "question": questions,
            "contexts": contexts,
            "evidences": evidences
        }
        dataset = Dataset.from_dict(data)
        
        # If sample
        if args.num_samples is not None:
            selected_indices = sampler.sample(range(len(dataset)), args.num_samples)
            dataset = dataset.select(selected_indices)
        
        # Perform evaluation
        results = await evaluate_dataset(
            dataset=dataset,
            llm=llm, 
            embeddings=embedding,
            detailed_output=args.detailed_output
        )
        
        all_results[question_type] = results
        print(f"\nResults for {question_type}:")
        for metric, score in results["average_scores"].items():
            counts = results["metric_counts"][metric]
            rendered = f"{score:.4f}" if score is not None else "FAILED"
            print(f"  {metric}: {rendered} ({counts['valid']}/{results['total_samples']})")
            
    # Save final results
    if args.output_file:
        print(f"\nSaving results to {args.output_file}...")
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        complete = not any(result["failed_samples"] for result in all_results.values())
        report = {
            "metadata": {
                "mode": args.mode,
                "model": args.model,
                "embedding_model": args.embedding_model,
                "input_file": str(args.data_file),
                "num_samples_per_question_type": args.num_samples,
                "sample_seed": SEED,
                "complete": complete,
            },
            "question_types": all_results,
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)

    if args.mode == "ollama":
        await llm.close()
    complete = not any(result["failed_samples"] for result in all_results.values())
    if not complete:
        raise SystemExit("Evaluation finished with failed metrics; inspect failed_samples in the report")
    print('\nEvaluation complete with no failed metrics.')

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add command-line arguments
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
        default="retrieval_results.json",
        help="Path to save evaluation results"
    )
    
    parser.add_argument(
        "--num_samples", 
        type=int,
        default=None,
        help="Number of samples per question type to evaluate (optional)"
    )

    parser.add_argument(
        "--detailed_output",
        action="store_true",
        help="Whether to include detailed output"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the main function
    asyncio.run(main(args))
