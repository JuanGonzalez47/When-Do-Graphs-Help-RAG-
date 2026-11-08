"""Rutas centrales del proyecto, calculadas desde la ubicación de este módulo."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # Sube desde research/src hasta la raíz del proyecto.
OFFICIAL = ROOT / "official" / "GraphRAG-Benchmark"  # Clon de referencia, separado del código de investigación.
DATASETS = OFFICIAL / "Datasets"  # Carpeta que contiene las preguntas y los textos.
QUESTIONS = DATASETS / "Questions"  # Archivos JSON de preguntas oficiales.
CORPUS = DATASETS / "Corpus"  # Textos fuente para retrieval y extracción.

RESEARCH = ROOT / "research"  # Paquete y datos propios de los experimentos.
CONFIGS = RESEARCH / "configs"  # Muestra fija y configuraciones.
GRAPHS = RESEARCH / "graphs"  # Grafos generados por corpus.
CACHE = RESEARCH / "cache"  # Resultados intermedios que evitan repetir trabajo costoso.
RESULTS = RESEARCH / "results"  # Reportes agregados y análisis.
OUTPUTS = RESULTS / "evaluation_raw"  # Predicciones y checkpoints sin agregar.

SAMPLE_FILE = CONFIGS / "questions_400.json"  # Muestra congelada de 400 preguntas.
SAMPLE_REPORT = RESULTS / "sampling_report.json"  # Auditoría de balance/digest de la muestra.
MANIFEST = RESULTS / "manifest.csv"  # Registro resumido de ejecuciones del pipeline.


def ensure_dirs() -> None:
    """Crea las carpetas que necesita el pipeline sin borrar datos existentes."""
    for path in (GRAPHS, CACHE, OUTPUTS, RESULTS):
        path.mkdir(parents=True, exist_ok=True)  # Crea también padres; no falla si ya existe.
    (OUTPUTS / "q1_q2" / "raw").mkdir(parents=True, exist_ok=True)  # JSONL incremental Q1/Q2.
    (OUTPUTS / "q1_q2" / "predictions").mkdir(parents=True, exist_ok=True)  # Predicciones completas Q1/Q2.
