#!/usr/bin/env python3
"""Consolida en formato largo las métricas disponibles de Q1, Q2, Q3 y Q4.

El CSV mantiene separadas las medias descriptivas, las diferencias pareadas y
los intervalos de confianza. También identifica reportes parciales y el origen
de cada fila para que la consolidación no oculte limitaciones experimentales.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "research" / "results"
Q1_Q2_REPORTS = RESULTS / "evaluation_q1_q2" / "q1q2_n50_final"
Q3_Q4_RESULTS = RESULTS / "evaluation_q3_q4"
OUTPUT = RESULTS / "results.csv"

FIELDS = (
    "experiment",
    "phase",
    "corpus",
    "question_type",
    "condition",
    "comparison",
    "budget_tokens",
    "metric",
    "n",
    "mean",
    "delta",
    "ci_lower",
    "ci_upper",
    "status",
    "source",
)

Q1_CONDITIONS = {"rag", "graph_content", "graph_full", "graph_rewired"}
Q2_CONDITIONS = {"rag", "graph_full", "graph_greedy", "graph_adaptive"}


def _read_json(path: Path) -> dict[str, Any]:
    """Carga un reporte JSON y exige un objeto en su nivel superior."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _number(value: Any) -> float | None:
    """Convierte a número solo valores finitos; los faltantes quedan vacíos."""
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _row(
    *,
    experiment: str,
    phase: str,
    corpus: str,
    question_type: str,
    condition: str,
    comparison: str = "",
    budget_tokens: int | str = "",
    metric: str,
    n: int | str = "",
    mean: float | None = None,
    delta: float | None = None,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
    status: str,
    source: str,
) -> dict[str, Any]:
    """Crea una fila con las columnas comunes y sin fabricar valores ausentes."""
    return {
        "experiment": experiment,
        "phase": phase,
        "corpus": corpus,
        "question_type": question_type,
        "condition": condition,
        "comparison": comparison,
        "budget_tokens": budget_tokens,
        "metric": metric,
        "n": n,
        "mean": mean,
        "delta": delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "status": status,
        "source": source,
    }


def _q1_q2_rows() -> list[dict[str, Any]]:
    """Extrae medias por tipo y totales ponderados de los reportes Q1/Q2 n=50."""
    rows: list[dict[str, Any]] = []
    if not Q1_Q2_REPORTS.is_dir():
        raise FileNotFoundError(f"Q1/Q2 reports not found: {Q1_Q2_REPORTS}")

    for path in sorted(Q1_Q2_REPORTS.glob("*.json")):
        # El nombre codifica corpus, condición, presupuesto y familia evaluada.
        parts = path.stem.split("_")
        if len(parts) < 4 or parts[0] not in {"novel", "medical"}:
            continue
        corpus, family = parts[0], parts[-1]
        try:
            budget = int(parts[-2])
        except ValueError:
            continue
        condition = "_".join(parts[1:-2])
        experiments = [
            experiment
            for experiment, conditions in (
                ("Q1", Q1_CONDITIONS),
                ("Q2", Q2_CONDITIONS),
            )
            if condition in conditions
        ]
        if not experiments:
            continue

        report = _read_json(path)
        metadata = report.get("metadata", {})
        complete = metadata.get("complete") is True
        status = "complete" if complete else "incomplete_or_unverified"
        total_by_metric: dict[str, list[tuple[float, int]]] = {}

        for question_type, group in report.get("question_types", {}).items():
            # Cada métrica puede tener menos observaciones válidas que el grupo.
            averages = group.get("average_scores", {})
            counts = group.get("metric_counts", {})
            for metric, raw_mean in averages.items():
                mean = _number(raw_mean)
                valid = counts.get(metric, {}).get("valid", 0)
                if mean is None or not isinstance(valid, int) or valid <= 0:
                    continue
                total_by_metric.setdefault(metric, []).append((mean, valid))
                for experiment in experiments:
                    rows.append(
                        _row(
                            experiment=experiment,
                            phase="final_n50",
                            corpus=corpus,
                            question_type=question_type,
                            condition=condition,
                            budget_tokens=budget,
                            metric=metric,
                            n=valid,
                            mean=mean,
                            status=status,
                            source=str(path.relative_to(ROOT)),
                        )
                    )

        # El promedio global pondera cada tipo por su número de juicios válidos.
        for metric, values in total_by_metric.items():
            n = sum(count for _, count in values)
            mean = sum(value * count for value, count in values) / n
            for experiment in experiments:
                rows.append(
                    _row(
                        experiment=experiment,
                        phase="final_n50",
                        corpus=corpus,
                        question_type="overall",
                        condition=condition,
                        budget_tokens=budget,
                        metric=metric,
                        n=n,
                        mean=mean,
                        status=status,
                        source=str(path.relative_to(ROOT)),
                    )
                )
    return rows


def _q3_rows() -> list[dict[str, Any]]:
    """Lee medias por condición y contrastes pareados del análisis Q3."""
    path = Q3_Q4_RESULTS / "q3_paired_analysis.json"
    report = _read_json(path)
    rows: list[dict[str, Any]] = []

    for metric, conditions in report.get("condition_means", {}).items():
        for condition, summary in conditions.items():
            for corpus in ("overall", "novel", "medical"):
                stats = summary.get(corpus, {})
                rows.append(
                    _row(
                        experiment="Q3",
                        phase="all",
                        corpus=corpus,
                        question_type="overall",
                        condition=condition,
                        budget_tokens=4096,
                        metric=metric,
                        n=stats.get("n", ""),
                        mean=_number(stats.get("mean")),
                        status="descriptive",
                        source=str(path.relative_to(ROOT)),
                    )
                )

    for metric, contrasts in report.get("primary_contrasts", {}).items():
        for label, stats in contrasts.items():
            # Las claves expresan graph_intact menos la condición comparadora.
            prefix = "graph_intact_minus_"
            if not label.startswith(prefix):
                raise ValueError(f"Unexpected Q3 contrast name: {label}")
            rows.append(
                _row(
                    experiment="Q3",
                    phase="paired",
                    corpus="overall",
                    question_type="overall",
                    condition="graph_intact",
                    comparison=label.removeprefix(prefix),
                    budget_tokens=4096,
                    metric=metric,
                    n=stats.get("n", ""),
                    delta=_number(stats.get("mean_difference")),
                    ci_lower=_number(stats.get("ci_95_percentile", [None, None])[0]),
                    ci_upper=_number(stats.get("ci_95_percentile", [None, None])[1]),
                    status=stats.get("interpretation", "paired"),
                    source=str(path.relative_to(ROOT)),
                )
            )
    return rows


def _q4_rows() -> list[dict[str, Any]]:
    """Exporta medias y diferencias de la pantalla exploratoria de 80 preguntas."""
    path = Q3_Q4_RESULTS / "q4_preliminary_analysis.json"
    report = _read_json(path)
    rows: list[dict[str, Any]] = []
    status = str(report.get("interpretation", "exploratory"))

    for condition, summary in report.get("configurations", {}).items():
        for source_metric, metric in (
            ("answer_correctness_mean", "answer_correctness"),
            ("evidence_recall_mean", "evidence_recall"),
            ("context_tokens_mean", "context_tokens"),
        ):
            rows.append(
                _row(
                    experiment="Q4",
                    phase="preliminary80",
                    corpus="overall",
                    question_type="overall",
                    condition=condition,
                    budget_tokens=4096,
                    metric=metric,
                    n=summary.get("valid_questions", ""),
                    mean=_number(summary.get(source_metric)),
                    status=status,
                    source=str(path.relative_to(ROOT)),
                )
            )

    for metric, candidates in report.get("paired_differences", {}).items():
        for condition, references in candidates.items():
            for comparison, stats in references.items():
                interval = stats.get("ci_95_percentile_equal_strata", [None, None])
                rows.append(
                    _row(
                        experiment="Q4",
                        phase="paired_preliminary80",
                        corpus="overall",
                        question_type="overall",
                        condition=condition,
                        comparison=comparison,
                        budget_tokens=4096,
                        metric=metric,
                        n=stats.get("n", ""),
                        delta=_number(stats.get("equal_stratum_mean_difference")),
                        ci_lower=_number(interval[0]),
                        ci_upper=_number(interval[1]),
                        status="exploratory_only_equal_stratum_bootstrap",
                        source=str(path.relative_to(ROOT)),
                    )
                )
    return rows


def main() -> int:
    """Consolida fuentes existentes y escribe un CSV en UTF-8 reproducible."""
    rows = _q1_q2_rows() + _q3_rows() + _q4_rows()
    if not rows:
        raise ValueError("No experiment results were found to export")

    # Ordena primero por pregunta de investigación y luego por celda/métrica.
    experiment_order = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}
    corpus_order = {"overall": 0, "novel": 1, "medical": 2}
    rows.sort(
        key=lambda row: (
            experiment_order.get(row["experiment"], 99),
            row["phase"],
            corpus_order.get(row["corpus"], 99),
            row["condition"],
            int(row["budget_tokens"]) if row["budget_tokens"] else 0,
            row["metric"],
            row["question_type"],
            row["comparison"],
        )
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
