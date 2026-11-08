#!/usr/bin/env python3
"""Resume resultados pareados de corrección de respuesta y recuperación de evidencia en Q3.

Usa las preguntas compartidas por las condiciones de grafo, remuestrea por corpus
y tipo de pregunta, y calcula diferencias medias pareadas e intervalos percentiles.
No aplica corrección por comparaciones múltiples.
"""

# Mantiene las anotaciones de tipos sin evaluarlas al importar el módulo.
from __future__ import annotations

# json lee los reportes y guarda el resultado; math filtra puntuaciones no finitas.
import json
import math
# sys permite añadir el paquete de investigación al path.
import sys
# defaultdict facilita acumular diferencias por estrato sin inicializar listas.
from collections import defaultdict
# Path construye las rutas a reportes y archivos de salida.
from pathlib import Path
# Any describe estructuras JSON con campos de tipos variados.
from typing import Any

# NumPy calcula medias, intervalos y muestras bootstrap eficientemente.
import numpy as np

# ROOT es la raíz del proyecto; así las rutas no dependen de dónde se corra el comando.
ROOT = Path(__file__).resolve().parents[1]
# Permite cargar funciones compartidas en research/src.
sys.path.insert(0, str(ROOT / "research"))

# La lista de IDs y tipos congelada define la población analizada.
from src.data import load_sample  

# Directorio donde el evaluador guardó resultados detallados por pregunta.
REPORTS = ROOT / "research" / "results" / "evaluation_q3_q4" / "reports"
# Destino del resumen pareado que generará este programa.
OUTPUT = ROOT / "research" / "results" / "evaluation_q3_q4" / "q3_paired_analysis.json"
# Condiciones evaluadas: cinco variantes de grafo y una referencia RAG.
CONDITIONS = (
    "graph_intact",
    "graph_nodes_only",
    "graph_hidden_predicates",
    "graph_permuted_predicates",
    "graph_rewired_topology",
    "rag_reference",
)
# Cada métrica se busca en el archivo de la familia correspondiente del evaluador.
METRICS = {
    "answer_correctness": "generation",
    "evidence_recall": "retrieval",
}
# Cada contraste resta la condición de control a la condición intacta.
CONTRASTS = (
    "graph_nodes_only",
    "graph_hidden_predicates",
    "graph_permuted_predicates",
    "graph_rewired_topology",
    "rag_reference",
)
# Número de réplicas aleatorias para aproximar la incertidumbre de la media.
BOOTSTRAPS = 10_000
# Semilla fija para que los intervalos se reproduzcan al repetir el análisis.
SEED = 42


def _read_metric(condition: str, metric: str, family: str) -> dict[str, dict[str, Any]]:
    """Lee puntuaciones finitas de un reporte y las organiza por ID de pregunta."""
    # El nombre combina condición, presupuesto y familia (generation/retrieval).
    path = REPORTS / f"q3_{condition}_4096_{family}.json"
    # Carga el JSON producido por el evaluador oficial.
    report = json.loads(path.read_text(encoding="utf-8"))
    # El diccionario resultante facilita emparejar la misma pregunta entre condiciones.
    values: dict[str, dict[str, Any]] = {}
    # Los reportes agrupan preguntas por tipo; dentro están las puntuaciones detalladas.
    for question_type, group in report["question_types"].items():
        # Algunas evaluaciones pueden haber fallado y no contener puntuación válida.
        for item in group.get("detailed", []):
            # Extrae la métrica solicitada si existe.
            value = item.get("metrics", {}).get(metric)
            # Ignora textos, valores ausentes, NaN e infinitos.
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                # Guarda puntuación y tipo para construir los estratos después.
                values[str(item["id"])] = {
                    "score": float(value),
                    "question_type": question_type,
                }
    # Devuelve solo observaciones utilizables para el análisis numérico.
    return values


def _mean(values: dict[str, dict[str, Any]], ids: set[str] | None = None) -> float | None:
    """Calcula una media descriptiva, opcionalmente limitada a un conjunto de IDs."""
    # Selecciona todos los valores o únicamente los IDs solicitados.
    selected = [
        item["score"]
        for record_id, item in values.items()
        if ids is None or record_id in ids
    ]
    # None indica que no había observaciones, evitando inventar una media cero.
    return float(np.mean(selected)) if selected else None


def _paired_bootstrap(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    corpus_by_id: dict[str, str],
    seed_offset: int,
) -> dict[str, Any]:
    """Calcula diferencias pareadas y un intervalo bootstrap por corpus y tipo."""
    # Solo se comparan preguntas con puntuación en ambas condiciones.
    shared = set(left) & set(right)
    # Sin pares no existe una comparación válida.
    if not shared:
        raise ValueError("No paired Q3 scores found")
    # Cada lista guarda diferencias de preguntas del mismo corpus y tipo.
    strata: dict[tuple[str, str], list[float]] = defaultdict(list)
    # Calcula para cada pregunta: puntuación izquierda menos puntuación derecha.
    for record_id in shared:
        left_item = left[record_id]
        right_item = right[record_id]
        if left_item["question_type"] != right_item["question_type"]:
            raise ValueError(f"Question type mismatch for {record_id}")
        # El tipo debe coincidir entre archivos; una discordancia implica datos mal emparejados.
        difference = left_item["score"] - right_item["score"]
        # El ID aporta el corpus y el reporte aporta el tipo para clasificar el par.
        strata[(corpus_by_id[record_id], left_item["question_type"])].append(difference)

    # n es el número de IDs compartidos que realmente entran al análisis.
    n = len(shared)
    # Cada contraste tiene semilla propia, pero su selección es reproducible.
    rng = np.random.default_rng(SEED + seed_offset)
    # Aquí se acumulará una media por cada réplica bootstrap.
    bootstrap_means = np.zeros(BOOTSTRAPS, dtype=np.float64)
    # Remuestrea con reemplazo dentro de cada estrato y conserva su peso en la muestra.
    for differences in strata.values():
        # Convierte las diferencias de este estrato a un arreglo numérico.
        values = np.asarray(differences, dtype=np.float64)
        # Cada fila elige n observaciones con reemplazo para una réplica.
        sampled = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
        # Pondera el promedio del estrato según su fracción del total de pares.
        bootstrap_means += values[sampled].mean(axis=1) * (len(values) / n)
    # Calcula nuevamente las diferencias observadas para la estimación puntual.
    differences = np.asarray(
        [left[record_id]["score"] - right[record_id]["score"] for record_id in shared],
        dtype=np.float64,
    )
    # Devuelve estimación, límites percentiles del 95% y tamaños de estratos.
    return {
        "n": n,  # Cantidad de pares efectivamente utilizados.
        "mean_difference": float(differences.mean()),  # Izquierda menos derecha.
        "ci_95_percentile": [
            # Percentiles 2,5 y 97,5 de las medias bootstrap.
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ],
        "bootstrap_resamples": BOOTSTRAPS,  # Número de réplicas para el intervalo.
        "strata": {
            # Tamaño de cada grupo corpus/tipo utilizado para re-muestrear.
            f"{corpus}/{question_type}": len(values)
            for (corpus, question_type), values in sorted(strata.items())
        },
    }


def main() -> int:
    """Carga reportes, calcula contrastes Q3 pareados y escribe el resumen JSON."""
    # Los registros determinan los tipos y permiten mapear ID a Novel o Medical.
    records = load_sample()["records"]
    # Mapa usado para comprobar que las condiciones comparan los mismos tipos.
    type_by_id = {str(record["id"]): str(record["question_type"]) for record in records}
    # El corpus se deriva del prefijo de ID establecido por el dataset.
    corpus_by_id = {
        str(record["id"]): (
            "novel" if str(record["id"]).startswith("Novel") else "medical"
        )
        for record in records
    }
    # Estos IDs no admitieron el intercambio de aristas exigido por el protocolo.
    expected = set(type_by_id) - {
        "Medical-682c2e93",
        "Novel-6c90555e",
        "Novel-95d8f71f",
    }
    # Estructura inicial del archivo final y documentación del conjunto pareado.
    result: dict[str, Any] = {
        "protocol": "Q3 pooled paired analysis; 4096-token budget",  # Contexto del cálculo guardado.
        "paired_graph_question_count": len(expected),  # Preguntas comunes en variantes de grafo.
        "excluded_unrewirable_ids": sorted(set(type_by_id) - expected),  # IDs sin rewiring válido.
        "primary_contrasts": {},  # Diferencias e intervalos por comparación.
        "condition_means": {},  # Medias descriptivas por condición y corpus.
    }

    # Ejecuta el mismo proceso para corrección y recuperación de evidencia.
    for metric_index, (metric, family) in enumerate(METRICS.items()):
        # Carga la puntuación de cada condición para esta métrica.
        by_condition = {
            condition: _read_metric(condition, metric, family)
            for condition in CONDITIONS
        }
        # Se calculan medias generales y por corpus para facilitar la descripción.
        metric_means = {}
        for condition, values in by_condition.items():
            # El tamaño puede variar si algunos juicios del evaluador fallaron.
            condition_result: dict[str, Any] = {
                "overall": {
                    "n": len(values),  # Observaciones con puntuación finita.
                    "mean": _mean(values),  # Media sin ponderación adicional.
                }
            }
            # Separa Novel y Medical sin cambiar el conjunto de preguntas de cada uno.
            for corpus in ("novel", "medical"):
                corpus_ids = {
                    record_id
                    for record_id in values
                    if corpus_by_id[record_id] == corpus
                }
                condition_result[corpus] = {
                    "n": len(corpus_ids),  # Cantidad de IDs de este corpus.
                    "mean": _mean(values, corpus_ids),  # Media solo de ese corpus.
                }
            # Asocia el resumen al nombre de su condición.
            metric_means[condition] = condition_result

        # Contrasta graph_intact contra cada una de las otras cinco condiciones.
        contrasts = {}
        for contrast_index, condition in enumerate(CONTRASTS):
            # El offset asegura flujos aleatorios reproducibles y distintos por contraste.
            pair = _paired_bootstrap(
                by_condition["graph_intact"],
                by_condition[condition],
                corpus_by_id,
                metric_index * 100 + contrast_index,
            )
            # RAG usa otra forma de recuperación; no es una ablación causal equivalente.
            pair["interpretation"] = (
                "external RAG baseline; not a content-matched causal contrast"
                if condition == "rag_reference"
                else "matched graph intervention"
            )
            # La clave deja claro qué resta se calculó y en qué dirección.
            contrasts[f"graph_intact_minus_{condition}"] = pair
        # Añade medias y contrastes para la métrica actual al documento final.
        result["condition_means"][metric] = metric_means
        result["primary_contrasts"][metric] = contrasts

    # Crea el directorio si el proyecto todavía no tiene resultados Q3.
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # Guarda valores numéricos estrictos: NaN no se permite en JSON estándar.
    OUTPUT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    # Informa dónde se guardó el análisis.
    print(f"Saved {OUTPUT}")
    # Presenta un resumen legible de medias e intervalos en la consola.
    for metric in METRICS:
        print(f"\n{metric}")
        for condition, summary in result["condition_means"][metric].items():
            overall = summary["overall"]
            print(f"  {condition:28s} mean={overall['mean']:.4f} n={overall['n']}")
        for name, summary in result["primary_contrasts"][metric].items():
            low, high = summary["ci_95_percentile"]
            print(
                f"  {name:42s} delta={summary['mean_difference']:+.4f} "
                f"95% CI [{low:+.4f}, {high:+.4f}] n={summary['n']}"
            )
    # Código cero comunica éxito a quien ejecute el script o una automatización.
    return 0


# Ejecuta el análisis solo si el archivo se llama como programa.
if __name__ == "__main__":
    # SystemExit propaga el código de retorno de main al sistema operativo.
    raise SystemExit(main())
