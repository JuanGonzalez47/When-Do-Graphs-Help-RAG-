#!/usr/bin/env python3
"""Resume el subconjunto exploratorio congelado de Q4 y lo compara con referencias Q1.

Exige puntuaciones completas por pregunta y tokens reales para los 80 IDs.
El bootstrap pareado se estratifica por corpus y tipo de pregunta. Es una
exploración preliminar, no una confirmación en el holdout de Q4.
"""

# Habilita anotaciones de tipos modernas sin evaluarlas durante la importación.
from __future__ import annotations

# json lee archivos de configuración/resultados; math descarta valores no finitos.
import json
import math
# sys permite localizar las funciones compartidas del proyecto.
import sys
# defaultdict crea automáticamente una lista al encontrar un estrato nuevo.
from collections import defaultdict
# Path permite formar rutas válidas en el sistema operativo.
from pathlib import Path
# Any representa el contenido flexible de los reportes JSON.
from typing import Any

# NumPy se usa para medias y remuestreo estadístico reproducible.
import numpy as np

# ROOT identifica la raíz aunque se ejecute el programa desde otra carpeta.
ROOT = Path(__file__).resolve().parents[1]
# Añade research/ al camino de importación para poder usar src.data.
sys.path.insert(0, str(ROOT / "research"))

# Recupera la misma muestra congelada usada para construir los experimentos.
from src.data import load_sample  

# Carpeta de reportes de evaluación completos.
REPORTS = ROOT / "research" / "results" / "evaluation_q3_q4" / "reports"
# Algunos resultados del cribado de 80 preguntas se guardaron aparte.
PRELIMINARY_REPORTS = REPORTS / "preliminary80"
# Archivo con los IDs seleccionados para la evaluación preliminar.
SAMPLE_PATH = ROOT / "research" / "configs" / "q4_preliminary_20261108.json"
# Archivo donde se guardará el resumen y las comparaciones de este script.
OUTPUT = (
    ROOT
    / "research"
    / "results"
    / "evaluation_q3_q4"
    / "q4_preliminary_analysis.json"
)
# Cantidad de réplicas para estimar la distribución bootstrap de las diferencias.
BOOTSTRAPS = 10_000
# Semilla fija para que el resultado aleatorio se pueda reproducir.
SEED = 42


def _read_report_values(
    condition: str,
    metric: str,
    family: str,
) -> dict[str, dict[str, Any]]:
    """Lee puntuaciones finitas desde el reporte completo o preliminar correcto."""
    # Las referencias Q1 de esta pantalla se escribieron en el subdirectorio preliminar.
    if condition.startswith("q4_reference_"):
        report_path = PRELIMINARY_REPORTS / f"{condition}_{family}.json"
    # Los nombres q4_graph_*_h* son condiciones de profundidad de Q4.
    elif condition.startswith("q4_graph_") and "_h" in condition:
        # Por defecto, esas condiciones usan reportes de la pantalla de 80 IDs.
        report_path = PRELIMINARY_REPORTS / f"{condition}_{family}.json"
        # Tres graph_full se evaluaron antes con el reporte completo de desarrollo.
        if condition.startswith(("q4_graph_full_h1_", "q4_graph_full_h2_", "q4_graph_full_h3_")):
            report_path = REPORTS / f"{condition}_{family}.json"
    else:
        # Para otros casos, toma el reporte general de evaluación.
        report_path = REPORTS / f"{condition}_{family}.json"
    # Lee el archivo seleccionado como objeto JSON.
    report = json.loads(report_path.read_text(encoding="utf-8"))
    # Guarda valores por ID para poder comparar la misma pregunta entre condiciones.
    values: dict[str, dict[str, Any]] = {}
    # Cada reporte organiza resultados detallados en grupos por tipo de pregunta.
    for question_type, group in report["question_types"].items():
        # "detailed" conserva el resultado individual de cada pregunta.
        for item in group.get("detailed", []):
            # Extrae la métrica deseada; puede no existir si falló el juez.
            score = item.get("metrics", {}).get(metric)
            # Acepta números finitos únicamente para cálculos estadísticos.
            if isinstance(score, (int, float)) and math.isfinite(float(score)):
                # Mantiene también el tipo de pregunta para validar el emparejamiento.
                values[str(item["id"])] = {
                    "score": float(score),
                    "question_type": question_type,
                }
    # Devuelve el mapa de puntuaciones pregunta -> valor y estrato.
    return values


def _prediction_context_tokens(name: str, selected_ids: set[str]) -> dict[str, int]:
    """Carga el total de tokens de contexto realmente generado para cada ID elegido."""
    # Los archivos de referencia Q1 se materializaron como JSON de entrada del evaluador.
    if name.startswith("q4_reference_"):
        path = ROOT / "research" / "results" / "evaluation_q3_q4" / "inputs" / (
            f"{name}_preliminary80.json"
        )
        # Lee las filas ya preparadas para el evaluador.
        rows = json.loads(path.read_text(encoding="utf-8"))
    else:
        # Las condiciones nuevas de Q4 guardan una predicción por línea en JSONL.
        path = (
            ROOT
            / "research"
            / "results"
            / "evaluation_raw"
            / "q3_q4"
            / "generation"
            / "q4"
            / "development"
            / f"{name}.jsonl"
        )
        # Convierte cada línea JSON no vacía a un registro Python.
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    # Se limita al subconjunto y se devuelve ID -> tokens reportados en generación.
    return {
        str(row["id"]): int(row["context_tokens"])
        for row in rows
        if str(row["id"]) in selected_ids
    }


def _stratified_bootstrap(
    candidate: dict[str, dict[str, Any]],
    reference: dict[str, dict[str, Any]],
    corpus_by_id: dict[str, str],
    seed_offset: int,
) -> dict[str, Any]:
    """Calcula diferencias pareadas con igual peso para cada estrato corpus/tipo."""
    # Empareja únicamente preguntas con una puntuación en ambas condiciones.
    shared = set(candidate) & set(reference)
    # Si no existen pares, no se puede estimar una diferencia.
    if not shared:
        raise ValueError("No paired Q4 metric values are available")
    # Acumula deltas por combinación de corpus y tipo de pregunta.
    strata: dict[tuple[str, str], list[float]] = defaultdict(list)
    # Para cada pregunta se resta referencia a candidata.
    for record_id in shared:
        # Los tipos deben coincidir para asegurar que la comparación sea válida.
        if candidate[record_id]["question_type"] != reference[record_id]["question_type"]:
            raise ValueError(f"Question type mismatch for {record_id}")
        # Guarda el delta dentro del estrato correspondiente.
        strata[
            (corpus_by_id[record_id], candidate[record_id]["question_type"])
        ].append(candidate[record_id]["score"] - reference[record_id]["score"])

    # Número de preguntas completas incluidas.
    n = len(shared)
    # Generador pseudoaleatorio con semilla fija más un offset por comparación.
    rng = np.random.default_rng(SEED + seed_offset)
    # Arreglo para almacenar el delta estimado en cada repetición.
    boot = np.zeros(BOOTSTRAPS, dtype=np.float64)
    # Total de estratos presentes; cada uno tendrá igual peso en el estimador.
    total_strata = len(strata)
    # Remuestrea con reemplazo dentro de cada estrato y promedia sus medias por igual.
    for differences in strata.values():
        # NumPy requiere arreglos numéricos para indexar y calcular promedios.
        array = np.asarray(differences, dtype=np.float64)
        # Una fila de índices equivale a una muestra bootstrap del estrato.
        sampled = rng.integers(0, len(array), size=(BOOTSTRAPS, len(array)))
        # Suma el promedio del estrato con peso 1/número de estratos.
        boot += array[sampled].mean(axis=1) / total_strata
    # Calcula el delta observado usando todos los pares compartidos.
    observed = np.asarray(
        [
            candidate[record_id]["score"] - reference[record_id]["score"]
            for record_id in shared
        ],
        dtype=np.float64,
    )
    # Devuelve tanto promedio simple como promedio por estratos y su intervalo del 95%.
    return {
        "n": n,  # Número de IDs que aparecen en ambas condiciones.
        "mean_difference": float(observed.mean()),  # Promedio simple candidato menos referencia.
        "equal_stratum_mean_difference": float(
            np.mean([np.mean(values) for values in strata.values()])
        ),  # Cada corpus/tipo tiene igual peso, independiente de su tamaño.
        "ci_95_percentile_equal_strata": [
            # Intervalo bootstrap percentil del 95% con la misma ponderación.
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "bootstrap_resamples": BOOTSTRAPS,  # Réplicas utilizadas para estimar incertidumbre.
    }


def main() -> int:
    """Valida los 80 IDs y calcula medias e intervalos frente a referencias Q1."""
    # Lee la configuración congelada del subconjunto preliminar.
    sample_data = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    # Un set facilita comprobar cobertura y evitar considerar un ID dos veces.
    selected_ids = set(sample_data["ids"])
    # Carga los 400 registros de la muestra experimental original.
    records = load_sample()["records"]
    # Indexa el conjunto original por ID para verificar pertenencia del subconjunto.
    record_by_id = {str(record["id"]): record for record in records}
    # Rechaza muestras incompletas o con IDs que no pertenecen a la muestra oficial.
    if len(selected_ids) != 80 or not selected_ids <= set(record_by_id):
        raise ValueError(f"Invalid preliminary Q4 sample in {SAMPLE_PATH}")
    # Determina el corpus de cada ID para usarlo como estrato estadístico.
    corpus_by_id = {
        record_id: ("novel" if record_id.startswith("Novel") else "medical")
        for record_id in selected_ids
    }

    # Incluye seis celdas que sí fueron parte del cribado y las dos referencias Q1.
    configurations = [
        *(f"q4_graph_full_h{depth}_4096" for depth in (1, 2, 3)),
        *(
            f"q4_graph_{method}_h{depth}_4096"
            for method in ("greedy", "adaptive")
            for depth in (1, 3, 6)
        ),
        "q4_reference_rag_4096",
        "q4_reference_graph_full_4096",
    ]
    # Prepara el formato del resultado e indica que la inferencia es exploratoria.
    result: dict[str, Any] = {
        "sample": str(SAMPLE_PATH.relative_to(ROOT)),  # Archivo que define los 80 IDs.
        "sample_size": len(selected_ids),  # Cantidad de preguntas del cribado.
        "sample_seed": sample_data["seed"],  # Semilla que generó el subconjunto.
        "interpretation": "exploratory only; not a confirmation of the preregistered threshold",  # No afirmar confirmación.
        "configurations": {},  # Medias de tokens, corrección y recall por celda.
        "paired_differences": {},  # Comparaciones de cada candidata con dos referencias.
    }
    # Acumula las puntuaciones detalladas por configuración para emparejarlas luego.
    score_data: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    # Lee todas las configuraciones antes de calcular comparaciones.
    for config in configurations:
        # Correctness viene de evaluación de generación; recall de retrieval.
        gen = _read_report_values(config, "answer_correctness", "generation")
        rec = _read_report_values(config, "evidence_recall", "retrieval")
        # Mantiene solo observaciones pertenecientes a los 80 IDs congelados.
        gen = {key: value for key, value in gen.items() if key in selected_ids}
        rec = {key: value for key, value in rec.items() if key in selected_ids}
        # Este análisis requiere cada una de las preguntas, no una muestra parcial.
        expected = selected_ids
        if set(gen) != expected or set(rec) != expected:
            raise ValueError(
                f"{config}: incomplete selected metric values "
                f"(correctness={len(gen)}, recall={len(rec)}, expected=80)"
            )
        # Recupera el número de tokens que se envió realmente en el contexto.
        tokens = _prediction_context_tokens(config, selected_ids)
        # Si faltan tokens, no se puede comparar la eficiencia de contexto.
        if set(tokens) != expected:
            raise ValueError(f"{config}: actual context token data does not cover all 80 IDs")
        score_data[config] = {
            "answer_correctness": gen,
            "evidence_recall": rec,
        }
        # Resume longitud media del contexto y calidad media en las 80 preguntas.
        result["configurations"][config] = {
            "context_tokens_mean": float(np.mean(list(tokens.values()))),  # Eficiencia de contexto observada.
            "answer_correctness_mean": float(
                np.mean([value["score"] for value in gen.values()])
            ),  # Media del juez de corrección.
            "evidence_recall_mean": float(
                np.mean([value["score"] for value in rec.values()])
            ),  # Media del juez de recuperación de evidencia.
            "valid_questions": 80,  # Se exige cobertura completa del subconjunto.
        }

    # Compara ambos resultados primarios (correctness y evidence recall).
    for metric_index, metric in enumerate(("answer_correctness", "evidence_recall")):
        # Cada métrica tendrá una sección independiente en el JSON.
        result["paired_differences"][metric] = {}
        # Las últimas dos configuraciones son las referencias y no son candidatas.
        for config_index, config in enumerate(configurations[:-2]):
            # Reserva resultados comparativos para la configuración actual.
            result["paired_differences"][metric][config] = {}
            # Toda candidata se compara con las referencias RAG y graph_full.
            for reference_index, reference_name in enumerate(
                ("q4_reference_rag_4096", "q4_reference_graph_full_4096")
            ):
                # El offset separa las semillas usadas por cada métrica/celda/referencia.
                result["paired_differences"][metric][config][reference_name] = (
                    _stratified_bootstrap(
                        score_data[config][metric],
                        score_data[reference_name][metric],
                        corpus_by_id,
                        metric_index * 100 + config_index * 2 + reference_index,
                    )
                )

    # Crea la carpeta de resultados si aún no existe.
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # Guarda resumen y comparaciones; allow_nan=False evita JSON con NaN inválido.
    OUTPUT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    # Indica dónde quedó el análisis generado.
    print(f"Saved {OUTPUT}")
    # Imprime por configuración los tokens y las medias de las dos métricas.
    for config, summary in result["configurations"].items():
        print(
            f"{config:36s} tokens={summary['context_tokens_mean']:.1f} "
            f"correct={summary['answer_correctness_mean']:.4f} "
            f"recall={summary['evidence_recall_mean']:.4f}"
        )
    # Código cero comunica al sistema operativo que el análisis finalizó bien.
    return 0


# Evita ejecutar el análisis cuando este archivo se importe desde otro módulo.
if __name__ == "__main__":
    # Devuelve el estado de main al shell.
    raise SystemExit(main())
