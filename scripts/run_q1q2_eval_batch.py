#!/usr/bin/env python3
"""Run and validate official generation/retrieval metrics for Q1/Q2 cells.

Predictions are produced by the research harness; this script prepares retrieval
evidence where needed, invokes the cloned benchmark evaluator, and verifies that
all expected per-question-type metric records completed successfully.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("novel", "medical")
METHODS = (
    "rag",
    "graph_content",
    "graph_full",
    "graph_rewired",
    "graph_greedy",
    "graph_adaptive",
)
QUESTION_TYPES = (
    "Fact Retrieval",
    "Complex Reasoning",
    "Contextual Summarize",
    "Creative Generation",
)


def report_is_complete(path: Path, num_samples: int, family: str) -> bool:
    """Check expected question groups, metric names, counts, and failed judgments."""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    metadata = report.get("metadata", {})
    groups = report.get("question_types", {})
    if not metadata.get("complete") or set(groups) != set(QUESTION_TYPES):
        return False
    expected_metrics = {
        "generation": {
            "Fact Retrieval": {"rouge_score", "answer_correctness"},
            "Complex Reasoning": {"rouge_score", "answer_correctness"},
            "Contextual Summarize": {"answer_correctness", "coverage_score"},
            "Creative Generation": {
                "answer_correctness",
                "coverage_score",
                "faithfulness",
            },
        },
        "retrieval": {
            question_type: {"context_relevancy", "evidence_recall"}
            for question_type in QUESTION_TYPES
        },
    }
    for question_type, result in groups.items():
        if result.get("total_samples") != num_samples or result.get("failed_samples"):
            return False
        metrics = result.get("metric_counts", {})
        if set(metrics) != expected_metrics[family][question_type]:
            return False
        for counts in metrics.values():
            if counts.get("valid") != num_samples or counts.get("failed") != 0:
                return False
    return True


def run(command: list[str], env: dict[str, str]) -> bool:
    """Print and execute one official evaluator subprocess from the project root."""
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode == 0


def main() -> int:
    """Evaluate every selected split/method/budget/family and report failures."""
    parser = argparse.ArgumentParser(
        description="Run and validate Q1/Q2 evaluation batches."
    )
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--budgets", nargs="+", type=int, choices=(1024, 4096), default=[1024, 4096])
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("generation", "retrieval"),
        default=["generation", "retrieval"],
    )
    args = parser.parse_args()
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    official = ROOT / "official" / "GraphRAG-Benchmark"
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(official), existing_pythonpath) if part
    )
    # Keep the official benchmark importable without installing it into the research environment.
    model = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
    failures: list[str] = []
    total = len(args.splits) * len(args.methods) * len(args.budgets) * len(args.families)
    completed = 0

    for split in args.splits:
        for method in args.methods:
            for budget in args.budgets:
                prediction = (
                    ROOT
                    / "research"
                    / "results"
                    / "evaluation_raw"
                    / "q1_q2"
                    / "predictions"
                    / split
                    / f"{method}_{budget}.json"
                )
                name = f"{split}_{method}_{budget}"
                if not prediction.is_file():
                    failures.append(f"MISSING {prediction}")
                    print(f"MISSING prediction: {prediction}", flush=True)
                    continue

                for family in args.families:
                    completed += 1
                    report_path = output_dir / f"{name}_{family}.json"
                    if report_is_complete(report_path, args.num_samples, family):
                        print(
                            f"[{completed}/{total}] already complete: {report_path.name}",
                            flush=True,
                        )
                        continue

                    if family == "retrieval":
                        # The official retrieval evaluator expects evidence as a string list.
                        prepared = prediction.with_name(
                            f"{prediction.stem}_eval{prediction.suffix}"
                        )
                        if not run(
                            [
                                sys.executable,
                                "-u",
                                str(ROOT / "scripts" / "prepare_eval_data.py"),
                                str(prediction),
                            ],
                            env,
                        ):
                            failures.append(f"PREPARE_FAIL {name}")
                            continue
                        data_file = prepared
                        module = "Evaluation.retrieval_eval"
                    else:
                        data_file = prediction
                        module = "Evaluation.generation_eval"

                    command = [
                        sys.executable,
                        "-u",
                        "-m",
                        module,
                        "--mode",
                        "ollama",
                        "--model",
                        model,
                        "--base_url",
                        "http://localhost:11434",
                        "--embedding_model",
                        "nomic-embed-text",
                        "--data_file",
                        str(data_file),
                        "--output_file",
                        str(report_path),
                        "--num_samples",
                        str(args.num_samples),
                    ]
                    print(
                        f"[{completed}/{total}] evaluating {name} ({family})",
                        flush=True,
                    )
                    process_ok = run(command, env)
                    if not process_ok or not report_is_complete(
                        report_path, args.num_samples, family
                    ):
                        failures.append(f"{family.upper()}_FAIL {name}")
                        print(
                            f"FAILED or incomplete report: {report_path}",
                            flush=True,
                        )
                    else:
                        print(f"PASS: {report_path.name}", flush=True)

    failure_path = output_dir / "failures.log"
    failure_path.write_text(
        "".join(f"{failure}\n" for failure in failures), encoding="utf-8"
    )
    expected_reports = total
    existing_reports = sum(
        1
        for path in output_dir.glob("*.json")
        if path.name.endswith(("_generation.json", "_retrieval.json"))
    )
    print(
        f"Batch finished: {existing_reports}/{expected_reports} reports; "
        f"{len(failures)} failures. Details: {failure_path}",
        flush=True,
    )
    return 1 if failures or existing_reports != expected_reports else 0


if __name__ == "__main__":
    raise SystemExit(main())
