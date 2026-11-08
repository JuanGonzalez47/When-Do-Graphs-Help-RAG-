#!/usr/bin/env python3
"""Preflight, prepare, and evaluate completed Q3/Q4 predictions.

The runner checks generated records against the frozen samples and protocol
before invoking the official evaluator in a subprocess. Evaluation inputs and
reports are kept separate from generation checkpoints and raw predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from src.data import load_sample

GENERATION_ROOT = ROOT / "research" / "results" / "evaluation_raw" / "q3_q4" / "generation"
EVALUATION_ROOT = ROOT / "research" / "results" / "evaluation_q3_q4"
INPUT_ROOT = EVALUATION_ROOT / "inputs"
REPORT_ROOT = EVALUATION_ROOT / "reports"
QUESTION_TYPES = (
    "Fact Retrieval",
    "Complex Reasoning",
    "Contextual Summarize",
    "Creative Generation",
)
Q3_CONDITIONS = (
    "graph_intact",
    "graph_nodes_only",
    "graph_hidden_predicates",
    "graph_permuted_predicates",
    "graph_rewired_topology",
    "rag_reference",
)
Q4_METHODS = ("graph_full", "graph_greedy", "graph_adaptive")
Q4_DEPTHS = tuple(range(1, 7))
Q3_EXCLUDED_IDS = {
    "Medical-682c2e93",
    "Novel-6c90555e",
    "Novel-95d8f71f",
}
Q4_PRELIMINARY_SPLIT = ROOT / "research" / "configs" / "q4_preliminary_20261108.json"
Q4_PRELIMINARY_SEED = 20261108


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL with line-numbered errors and require object records."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _prediction_path(stage: str, name: str) -> Path:
    """Resolve the checkpoint file for a Q3 or Q4 generation cell."""
    if stage == "q3":
        return GENERATION_ROOT / "q3" / "all" / f"{name}.jsonl"
    return GENERATION_ROOT / "q4" / "development" / f"{name}.jsonl"


def _required_fields(row: dict[str, Any], path: Path) -> None:
    """Reject predictions that do not match the evaluator and experiment schema."""
    for field in ("id", "question", "question_type", "generated_answer", "ground_truth"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f"{path}: {row.get('id', '<unknown>')} has invalid {field}")
    if row["question_type"] not in QUESTION_TYPES:
        raise ValueError(f"{path}: unsupported question type {row['question_type']!r}")
    contexts = row.get("context")
    if not isinstance(contexts, list) or not all(isinstance(item, str) for item in contexts):
        raise ValueError(f"{path}: {row['id']} has invalid context list")
    if not contexts:
        raise ValueError(f"{path}: {row['id']} has empty context")
    metadata = row.get("experiment_metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError(f"{path}: {row['id']} has invalid experiment metadata")
    model = metadata.get("model") if isinstance(metadata, dict) else row.get("model")
    if model != "qwen2.5:3b":
        raise ValueError(f"{path}: {row['id']} has unexpected model {model!r}")
    if isinstance(metadata, dict) and metadata.get("temperature") != 0.0:
        raise ValueError(f"{path}: {row['id']} has unexpected temperature")
    context_tokens = row.get("context_tokens")
    if not isinstance(context_tokens, int) or not 0 <= context_tokens <= 4096:
        raise ValueError(f"{path}: {row['id']} has invalid context token count")


def _load_cell(stage: str, name: str) -> dict[str, dict[str, Any]]:
    """Load one prediction cell keyed by unique question ID."""
    path = _prediction_path(stage, name)
    if not path.is_file():
        raise FileNotFoundError(f"Missing prediction checkpoint: {path}")
    rows = _read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        _required_fields(row, path)
        record_id = row["id"]
        if record_id in by_id:
            raise ValueError(f"{path}: duplicate question ID {record_id}")
        by_id[record_id] = row
    return by_id


def _read_error_ids(name: str) -> set[str]:
    """Read known Q3 rewiring failures and reject unexpected generation errors."""
    path = GENERATION_ROOT / "q3" / "all" / f"{name}.errors.jsonl"
    if not path.exists():
        return set()
    errors = _read_jsonl(path)
    for error in errors:
        if "id" not in error or "error" not in error:
            raise ValueError(f"Malformed error record in {path}")
        if "RewireUnavailableError" not in error["error"]:
            raise ValueError(f"Unexpected generation error in {path}: {error}")
    return {str(error["id"]) for error in errors}


def _normalize_evidence(value: Any, record_id: str) -> list[str]:
    """Normalize evaluator evidence while rejecting blank or malformed evidence."""
    if isinstance(value, list):
        if not all(isinstance(item, str) and item.strip() for item in value):
            raise ValueError(f"{record_id}: evidence list has blank or non-string values")
        return [item.strip() for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{record_id}: evidence is empty")
        if ";" in text:
            result = [part.strip() for part in text.split(";") if part.strip()]
            if result:
                return result
        return [text]
    raise ValueError(f"{record_id}: evidence must be a string or list of strings")


def _validate_q3(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Require paired Q3 graph outputs and verify the three documented exclusions."""
    cells = {
        condition: _load_cell("q3", f"q3_{condition}_4096")
        for condition in Q3_CONDITIONS
    }
    expected = {str(record["id"]) for record in records}
    for condition, cell in cells.items():
        if not set(cell) <= expected:
            raise ValueError(f"Q3 {condition}: unexpected IDs {sorted(set(cell) - expected)}")

    graph_conditions = Q3_CONDITIONS[:-1]
    graph_sets = [set(cells[condition]) for condition in graph_conditions]
    # Pair only IDs present in every graph intervention; RAG remains the 400-ID baseline.
    common = set.intersection(*graph_sets)
    error_sets = {
        condition: _read_error_ids(f"q3_{condition}_4096")
        for condition in graph_conditions
    }
    if any(error_ids != Q3_EXCLUDED_IDS for error_ids in error_sets.values()):
        raise ValueError(f"Unexpected Q3 error IDs: {error_sets}")
    if common != expected - Q3_EXCLUDED_IDS:
        raise ValueError(
            f"Q3 common graph IDs do not match the 3 documented exclusions "
            f"(found {len(common)})"
        )
    if set(cells["rag_reference"]) != expected:
        raise ValueError("Q3 RAG reference must contain all 400 frozen questions")

    rows_by_id = {str(record["id"]): record for record in records}
    q3_records = []
    for record_id in sorted(common):
        graph_rows = [cells[condition][record_id] for condition in graph_conditions]
        token_counts = {row["context_tokens"] for row in graph_rows}
        matched_counts = {
            row["experiment_metadata"].get("matched_context_tokens")
            for row in graph_rows
        }
        # Exact token matching and equal hop selection are required for Q3 comparisons.
        if len(token_counts) != 1 or len(matched_counts) != 1:
            raise ValueError(f"Q3 graph contexts are not token-matched for {record_id}")
        if token_counts != matched_counts:
            raise ValueError(f"Q3 matched-token metadata disagrees for {record_id}")
        if any(row["experiment_metadata"].get("selected_hops") != 2 for row in graph_rows):
            raise ValueError(f"Q3 contexts do not share the agreed two-hop selection for {record_id}")
        rewired = cells["graph_rewired_topology"][record_id]["experiment_metadata"]
        if not isinstance(rewired.get("rewire_swaps"), int) or rewired["rewire_swaps"] <= 0:
            raise ValueError(f"Q3 rewired condition did not change topology for {record_id}")
        permuted = cells["graph_permuted_predicates"][record_id]["experiment_metadata"]
        if not isinstance(permuted.get("predicate_bundles_changed"), int) or permuted[
            "predicate_bundles_changed"
        ] <= 0:
            raise ValueError(f"Q3 permuted condition did not change predicate bindings for {record_id}")
        q3_records.append(rows_by_id[record_id])
    return q3_records


def _validate_q4(records: list[dict[str, Any]], split_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Validate all twelve Q4 development cells against the frozen 320-ID split."""
    split = json.loads(split_path.read_text(encoding="utf-8"))
    expected_ids = set(split.get("development_ids", []))
    if len(expected_ids) != 320 or len(split.get("confirmation_ids", [])) != 80:
        raise ValueError("Frozen Q4 split must contain 320 development and 80 confirmation IDs")
    sample_ids = {str(record["id"]) for record in records}
    if expected_ids | set(split["confirmation_ids"]) != sample_ids:
        raise ValueError("Q4 split does not match the frozen 400-question sample")
    if expected_ids & set(split["confirmation_ids"]):
        raise ValueError("Q4 development and confirmation IDs overlap")
    by_id = {str(record["id"]): record for record in records}
    for phase, ids, expected_per_cell in (
        ("development", expected_ids, 40),
        ("confirmation", set(split["confirmation_ids"]), 10),
    ):
        cell_counts = Counter(
            (
                "novel" if record_id.startswith("Novel") else "medical",
                str(by_id[record_id]["question_type"]),
            )
            for record_id in ids
        )
        if any(
            cell_counts[(corpus, question_type)] != expected_per_cell
            for corpus in ("novel", "medical")
            for question_type in QUESTION_TYPES
        ):
            raise ValueError(f"Q4 {phase} split is not stratified as specified")

    cells: dict[str, list[dict[str, Any]]] = {}
    for method in Q4_METHODS:
        for depth in Q4_DEPTHS:
            name = f"q4_{method}_h{depth}_4096"
            by_id = _load_cell("q4", name)
            if set(by_id) != expected_ids:
                raise ValueError(
                    f"{name}: expected exactly the 320 development IDs, found {len(by_id)}"
                )
            for row in by_id.values():
                metadata = row["experiment_metadata"]
                if metadata.get("depth") != depth or metadata.get("condition") != method:
                    raise ValueError(f"{name}: inconsistent method/depth metadata for {row['id']}")
                if metadata.get("requested_budget") != 4096:
                    raise ValueError(f"{name}: inconsistent token budget for {row['id']}")
                if method == "graph_adaptive" and metadata.get("max_hops") != depth:
                    raise ValueError(f"{name}: inconsistent adaptive depth for {row['id']}")
            cells[name] = [by_id[record_id] for record_id in sorted(expected_ids)]
    return cells


def _validate_q1_references(
    records: list[dict[str, Any]],
    development_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    """Validate Q1 RAG/graph-full references and select their Q4 development rows."""
    expected_by_id = {str(record["id"]): record for record in records}
    references: dict[str, list[dict[str, Any]]] = {}
    for condition in ("rag", "graph_full"):
        by_id: dict[str, dict[str, Any]] = {}
        for corpus in ("novel", "medical"):
            path = (
                ROOT
                / "research"
                / "results"
                / "evaluation_raw"
                / "q1_q2"
                / "predictions"
                / corpus
                / f"{condition}_4096.json"
            )
            rows = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(rows, list) or len(rows) != 200:
                raise ValueError(f"Q1 reference {path} must contain 200 predictions")
            for row in rows:
                _required_fields(row, path)
                if row.get("condition") != condition or row.get("budget") != 4096:
                    raise ValueError(f"Q1 reference {path}: mismatched condition or budget")
                record_id = row["id"]
                if record_id in by_id:
                    raise ValueError(f"Q1 reference {condition}: duplicate ID {record_id}")
                frozen = expected_by_id.get(record_id)
                if frozen is None or any(
                    row[field] != frozen[source_field]
                    for field, source_field in (
                        ("question", "question"),
                        ("question_type", "question_type"),
                        ("ground_truth", "answer"),
                    )
                ):
                    raise ValueError(
                        f"Q1 reference {condition}: frozen sample mismatch for {record_id}"
                    )
                if row.get("model") != "qwen2.5:3b":
                    raise ValueError(
                        f"Q1 reference {condition}: unexpected model for {record_id}"
                    )
                if condition == "graph_full" and row.get("condition_metadata", {}).get("hops") != 2:
                    raise ValueError(
                        f"Q1 graph_full reference does not use two hops for {record_id}"
                    )
                by_id[record_id] = row
        if set(by_id) != set(expected_by_id):
            raise ValueError(f"Q1 reference {condition} does not cover the frozen 400 IDs")
        references[condition] = [
            by_id[record_id] for record_id in sorted(development_ids)
        ]
    return references


def _q4_preliminary_ids(
    records: list[dict[str, Any]],
    development_ids: set[str],
    samples_per_stratum: int,
) -> set[str]:
    """Load or freeze a balanced exploratory Q4 subset within development IDs."""
    if samples_per_stratum <= 0:
        raise ValueError("--q4-samples-per-stratum must be positive")
    by_id = {str(record["id"]): record for record in records}
    if Q4_PRELIMINARY_SPLIT.exists():
        saved = json.loads(Q4_PRELIMINARY_SPLIT.read_text(encoding="utf-8"))
        if (
            saved.get("seed") != Q4_PRELIMINARY_SEED
            or saved.get("samples_per_corpus_question_type") != samples_per_stratum
        ):
            raise ValueError(f"Incompatible frozen Q4 preliminary split: {Q4_PRELIMINARY_SPLIT}")
        selected = set(saved.get("ids", []))
    else:
        grouped: dict[tuple[str, str], list[str]] = {}
        for record_id in development_ids:
            row = by_id[record_id]
            corpus = "novel" if record_id.startswith("Novel") else "medical"
            grouped.setdefault((corpus, str(row["question_type"])), []).append(record_id)
        selected = set()
        for corpus in ("novel", "medical"):
            for question_type in QUESTION_TYPES:
                ids = sorted(grouped.get((corpus, question_type), []))
                if len(ids) < samples_per_stratum:
                    raise ValueError(
                        f"Not enough Q4 development questions in {corpus}/{question_type}"
                    )
                seed_material = f"{Q4_PRELIMINARY_SEED}:{corpus}:{question_type}"
                seed = int.from_bytes(
                    hashlib.sha256(seed_material.encode()).digest()[:8], "big"
                )
                selected.update(random.Random(seed).sample(ids, samples_per_stratum))
        Q4_PRELIMINARY_SPLIT.parent.mkdir(parents=True, exist_ok=True)
        Q4_PRELIMINARY_SPLIT.write_text(
            json.dumps(
                {
                    "seed": Q4_PRELIMINARY_SEED,
                    "source_split": str(
                        Path("research/configs/q4_split_2026.json")
                    ),
                    "samples_per_corpus_question_type": samples_per_stratum,
                    "counts": {"total": len(selected)},
                    "ids": sorted(selected),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    expected_count = samples_per_stratum * len(QUESTION_TYPES) * 2
    if (
        len(selected) != expected_count
        or not selected <= development_ids
        or selected & (set(str(record["id"]) for record in records) - development_ids)
    ):
        raise ValueError(f"Invalid frozen Q4 preliminary sample: {Q4_PRELIMINARY_SPLIT}")
    counts = Counter(
        (
            "novel" if record_id.startswith("Novel") else "medical",
            str(by_id[record_id]["question_type"]),
        )
        for record_id in selected
    )
    if any(
        counts[(corpus, question_type)] != samples_per_stratum
        for corpus in ("novel", "medical")
        for question_type in QUESTION_TYPES
    ):
        raise ValueError(f"Q4 preliminary sample is not stratified: {Q4_PRELIMINARY_SPLIT}")
    return selected


def _write_input(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write evaluator-ready predictions with normalized evidence fields."""
    prepared = []
    for row in rows:
        item = dict(row)
        item["evidence"] = _normalize_evidence(row.get("evidence"), row["id"])
        prepared.append(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(prepared, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _report_complete(
    path: Path,
    counts_by_type: Counter[str],
    sample_count: int | None,
) -> bool:
    """Return whether a metric report contains every expected valid detailed sample."""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    metadata = report.get("metadata", {})
    groups = report.get("question_types", {})
    if not metadata.get("complete") or set(groups) != set(counts_by_type):
        return False
    for question_type, expected_count in counts_by_type.items():
        result = groups[question_type]
        expected = sample_count if sample_count is not None else expected_count
        if (
            result.get("total_samples") != expected
            or result.get("failed_samples")
            or len(result.get("detailed", [])) != expected
        ):
            return False
        for metric_count in result.get("metric_counts", {}).values():
            if metric_count.get("valid") != expected or metric_count.get("failed") != 0:
                return False
    return True


def _run_cell(
    name: str,
    family: str,
    input_path: Path,
    output_dir: Path,
    model: str,
    sample_count: int | None,
    env: dict[str, str],
) -> bool:
    """Run one official metric family unless a full report already exists."""
    report_path = output_dir / f"{name}_{family}.json"
    counts_by_type = Counter(
        row["question_type"] for row in json.loads(input_path.read_text(encoding="utf-8"))
    )
    if sample_count is None and _report_complete(report_path, counts_by_type, None):
        print(f"SKIP complete: {report_path}", flush=True)
        return True

    module = "Evaluation.generation_eval" if family == "generation" else "Evaluation.retrieval_eval"
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
        str(input_path),
        "--output_file",
        str(report_path),
        "--detailed_output",
    ]
    if sample_count is not None:
        command.extend(["--num_samples", str(sample_count)])
    print("+ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if result.returncode != 0:
        print(f"FAILED: {report_path}", flush=True)
        return False
    if sample_count is None and not _report_complete(report_path, counts_by_type, None):
        print(f"FAILED or incomplete report: {report_path}", flush=True)
        return False
    print(f"PASS: {report_path}", flush=True)
    return True


def main() -> int:
    """Preflight predictions, prepare aligned inputs, then run requested evaluations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--pilot-samples", type=int, default=1)
    parser.add_argument("--stages", nargs="+", choices=("q3", "q4"), default=("q3", "q4"))
    parser.add_argument("--q4-methods", nargs="+", choices=Q4_METHODS, default=Q4_METHODS)
    parser.add_argument("--q4-depths", nargs="+", type=int, choices=Q4_DEPTHS, default=Q4_DEPTHS)
    parser.add_argument(
        "--q4-samples-per-stratum",
        type=int,
        help="Select and freeze this many development IDs per corpus/question type.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPORT_ROOT)
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "qwen2.5:3b"))
    args = parser.parse_args()
    if args.pilot_samples <= 0:
        parser.error("--pilot-samples must be positive")

    # Validate all generated cells and references before starting any expensive judge calls.
    records = list(load_sample()["records"])
    q3_records = _validate_q3(records)
    split_path = ROOT / "research" / "configs" / "q4_split_2026.json"
    q4_cells = _validate_q4(records, split_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    development_ids = set(split["development_ids"])
    references = _validate_q1_references(records, development_ids)
    q4_sample_ids = (
        _q4_preliminary_ids(
            records, development_ids, args.q4_samples_per_stratum
        )
        if args.q4_samples_per_stratum is not None
        else development_ids
    )
    print(
        f"Preflight passed: Q3 paired IDs={len(q3_records)}/400 "
        f"(excluded={sorted(Q3_EXCLUDED_IDS)}); Q4 cells={len(q4_cells)} x 320; "
        "Q1 references=RAG + graph_full x 320",
        flush=True,
    )
    if args.preflight_only:
        return 0

    input_dir = INPUT_ROOT
    q3_input_paths = {}
    for condition in Q3_CONDITIONS if "q3" in args.stages else ():
        name = f"q3_{condition}_4096"
        cell = _load_cell("q3", name)
        rows = [cell[str(row["id"])] for row in q3_records]
        path = input_dir / f"{name}_paired397.json"
        _write_input(path, rows)
        q3_input_paths[name] = path
    q4_input_paths = {}
    for name, rows in q4_cells.items():
        method = name.removeprefix("q4_").split("_h", 1)[0]
        depth = int(name.split("_h", 1)[1].split("_", 1)[0])
        if method not in args.q4_methods or depth not in args.q4_depths:
            continue
        if "q4" not in args.stages:
            continue
        selected_rows = [row for row in rows if row["id"] in q4_sample_ids]
        suffix = (
            f"_preliminary{len(q4_sample_ids)}"
            if args.q4_samples_per_stratum is not None
            else ""
        )
        path = input_dir / f"{name}{suffix}.json"
        _write_input(path, selected_rows)
        q4_input_paths[name] = path
    reference_input_paths = {}
    for condition, rows in references.items():
        if "q4" not in args.stages:
            continue
        name = f"q4_reference_{condition}_4096"
        selected_rows = [row for row in rows if row["id"] in q4_sample_ids]
        suffix = (
            f"_preliminary{len(q4_sample_ids)}"
            if args.q4_samples_per_stratum is not None
            else ""
        )
        path = input_dir / f"{name}{suffix}.json"
        _write_input(path, selected_rows)
        reference_input_paths[name] = path
    print(
        f"Prepared evaluator inputs: {len(q3_input_paths)} Q3 cells x {len(q3_records)} "
        f"and {len(q4_input_paths)} Q4 cells + {len(reference_input_paths)} "
        f"Q1 references x {len(q4_sample_ids)}",
        flush=True,
    )
    if args.pilot_only:
        # The pilot mode evaluates one small Q4 cell and writes to a separate subfolder.
        q4_name = "q4_graph_full_h1_4096"
        cell_names = [q4_name]
        output_dir = args.output_dir / "pilot"
        sample_count = args.pilot_samples
    else:
        cell_names = list(q3_input_paths) + list(q4_input_paths) + list(reference_input_paths)
        output_dir = args.output_dir
        sample_count = None
    output_dir.mkdir(parents=True, exist_ok=True)

    official = ROOT / "official" / "GraphRAG-Benchmark"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(official), env.get("PYTHONPATH", ""))
        if part
    )
    failures = []
    total_runs = len(cell_names) * 2
    completed = 0
    for name in cell_names:
        input_path = q3_input_paths.get(
            name,
            q4_input_paths.get(name, reference_input_paths.get(name)),
        )
        if input_path is None:
            raise RuntimeError(f"No prepared input for {name}")
        for family in ("generation", "retrieval"):
            completed += 1
            print(f"[{completed}/{total_runs}] {name} ({family})", flush=True)
            if not _run_cell(
                name,
                family,
                input_path,
                output_dir,
                args.model,
                sample_count,
                env,
            ):
                failures.append(f"{name}_{family}")
    failure_path = output_dir / "failures.log"
    failure_path.write_text("".join(f"{failure}\n" for failure in failures), encoding="utf-8")
    print(
        f"Evaluation batch finished: {total_runs - len(failures)}/{total_runs} "
        f"successful; failures={len(failures)}; details={failure_path}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
