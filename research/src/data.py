"""Carga de datos, validación y construcción de la muestra congelada."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .io import read_json, sha256_text
from .paths import CORPUS, QUESTIONS, SAMPLE_FILE

LEVELS = [  # Los cuatro estratos de preguntas que deben aparecer balanceados.
    "Fact Retrieval",
    "Complex Reasoning",
    "Contextual Summarize",
    "Creative Generation",
]

SPLITS = ("novel", "medical")  # Los dos corpus incluidos en el benchmark.

QUESTION_FILES = {  # Archivos oficiales que contienen preguntas y respuestas.
    "novel": QUESTIONS / "novel_questions.json",
    "medical": QUESTIONS / "medical_questions.json",
}

CORPUS_FILES = {  # Textos fuente para construir chunks y grafos.
    "novel": CORPUS / "novel.json",
    "medical": CORPUS / "medical.json",
}

EVIDENCE_FIELDS = {  # Nombre que usa cada dataset para describir la evidencia esperada.
    "novel": "evidence_triple",
    "medical": "evidence_relations",
}


def load_questions(split: str, dedupe: bool = True) -> list[dict[str, Any]]:
    """Carga las preguntas de un corpus y, opcionalmente, elimina IDs repetidos."""
    if split not in QUESTION_FILES:
        raise ValueError(f"Unknown split: {split}")  # Falla explícitamente ante un corpus desconocido.
    data = read_json(QUESTION_FILES[split])  # Abre el JSON oficial asignado al corpus.
    if not isinstance(data, list):
        raise TypeError(f"Expected a list of questions for {split}")  # Verifica el esquema esperado.
    if not dedupe:
        return data  # Permite inspeccionar los datos originales sin filtrar duplicados.
    seen: set[str] = set()  # IDs que ya aparecieron en el archivo.
    rows: list[dict[str, Any]] = []  # Filas únicas que se devolverán.
    for row in data:
        qid = str(row["id"])  # Normaliza el tipo para comparar IDs consistentes.
        if qid in seen:
            continue  # Conserva la primera aparición del ID duplicado.
        seen.add(qid)  # Marca el ID para detectar repeticiones posteriores.
        rows.append(row)  # Mantiene intactos los demás campos de la pregunta.
    return rows  # Entrega filas únicas en su orden original.


def load_corpus(split: str) -> dict[str, str]:
    """Carga textos fuente y devuelve un mapa nombre de fuente -> contenido."""
    if split not in CORPUS_FILES:
        raise ValueError(f"Unknown split: {split}")  # No sustituye silenciosamente un corpus inválido.
    raw = read_json(CORPUS_FILES[split])  # Lee el archivo de corpus seleccionado.
    if isinstance(raw, list):
        pairs = [(str(row["corpus_name"]), str(row["context"])) for row in raw]  # Formato con varias fuentes.
    elif isinstance(raw, dict):
        pairs = [(str(raw["corpus_name"]), str(raw["context"]))]  # También acepta una fuente como objeto único.
    else:
        raise TypeError(f"Unexpected corpus format for {split}: {type(raw)}")  # Rechaza estructuras no soportadas.
    return dict(pairs)  # Facilita recuperar el texto por nombre de fuente.


def load_sample(path: str | Path = SAMPLE_FILE) -> dict[str, Any]:
    """Lee la muestra congelada y comprueba que contenga su lista de registros."""
    data = read_json(path)  # La ruta puede ser la oficial o una copia indicada por el usuario.
    if not isinstance(data, dict) or "records" not in data:
        raise ValueError("Sample file must contain a records list")  # Evita usar una muestra con forma inesperada.
    return data  # Devuelve también digest, semilla y metadatos guardados en el archivo.


def sample_records(path: str | Path = SAMPLE_FILE) -> list[dict[str, Any]]:
    """Obtiene únicamente la lista de preguntas de la muestra congelada."""
    return list(load_sample(path)["records"])  # Copia la secuencia para devolver una lista mutable independiente.


def group_by_source(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Agrupa preguntas por documento fuente para muestrear sin concentrarse."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)  # Crea una lista al encontrar una fuente nueva.
    for record in records:
        grouped[str(record["source"])].append(record)  # Añade cada pregunta al grupo de su documento.
    return dict(grouped)  # Convierte el defaultdict a un diccionario ordinario.


def evidence_field(split: str) -> str:
    """Devuelve el nombre de campo donde el corpus guarda su evidencia."""
    return EVIDENCE_FIELDS[split]  # La clave debe corresponder a uno de los splits declarados.


def parse_levels(records: Iterable[dict[str, Any]]) -> Counter:
    """Cuenta cuántas preguntas hay de cada tipo."""
    return Counter(str(record["question_type"]) for record in records)  # Útil para auditar el balance.


def validate_sample(
    records: list[dict[str, Any]],
    per_level: int = 50,
    levels: list[str] | None = None,
) -> dict[str, Any]:
    """Exige IDs únicos y el conteo configurado por tipo y corpus."""
    levels = levels or LEVELS  # Si no se indican tipos, utiliza los cuatro oficiales.
    if len(records) != 2 * len(levels) * per_level:
        raise ValueError(f"Expected {2 * len(levels) * per_level} records, got {len(records)}")  # Total de corpus × tipos × cuota.

    ids = [str(record["id"]) for record in records]  # Convierte todos los identificadores al mismo tipo.
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question IDs in sample")  # Un ID repetido sesgaría las comparaciones.

    report: dict[str, Any] = {"total": len(records), "splits": {}}  # Aquí se construye el reporte de balance.
    for split in SPLITS:
        # Los prefijos identifican a qué corpus pertenece cada ID del benchmark.
        subset = [record for record in records if str(record["id"]).startswith("Novel" if split == "novel" else "Medical")]
        counts = Counter(str(record["question_type"]) for record in subset)  # Cuenta tipos dentro del corpus actual.
        if len(subset) != len(levels) * per_level:
            raise ValueError(f"{split}: expected {len(levels) * per_level} records, got {len(subset)}")  # Rechaza particiones incompletas.
        for level in levels:
            if counts.get(level, 0) != per_level:
                raise ValueError(f"{split}/{level}: expected {per_level}, got {counts.get(level, 0)}")  # Cada estrato debe tener la cuota exacta.
        report["splits"][split] = {
            "total": len(subset),  # Cantidad de preguntas de este corpus.
            "by_level": {level: counts.get(level, 0) for level in levels},  # Conteos por tipo.
            "sources": len({str(record["source"]) for record in subset}),  # Diversidad de documentos fuente.
        }
    return report  # Devuelve conteos verificables sin modificar la muestra.


def _sample_level(
    questions: list[dict[str, Any]],
    per_level: int,
    sources: list[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Elige n preguntas alternando entre fuentes dentro de un tipo."""
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)  # Candidatas agrupadas por documento.
    for row in questions:
        by_source[str(row["source"])].append(row)  # Mantiene juntas preguntas del mismo documento.
    for source in by_source:
        by_source[source].sort(key=lambda row: str(row["id"]))  # Orden determinista antes del azar.
        rng.shuffle(by_source[source])  # Mezcla candidatos de esa fuente usando la semilla compartida.

    order = sorted(sources)  # Fija el orden inicial para no depender de hashes de diccionario.
    rng.shuffle(order)  # Aleatoriza qué fuente aporta primero en cada ronda.
    cursor = {source: 0 for source in order}  # Próximo candidato no usado por fuente.
    chosen: list[dict[str, Any]] = []  # Preguntas seleccionadas para el estrato actual.
    while len(chosen) < per_level:
        progressed = False  # Detecta si quedó algún candidato disponible en esta ronda.
        for source in order:
            if len(chosen) >= per_level:
                break  # No selecciona más de la cuota solicitada.
            bucket = by_source.get(source, [])  # Candidatos de la fuente, o lista vacía.
            index = cursor[source]  # Lee el próximo candidato de esa fuente.
            if index < len(bucket):
                chosen.append(bucket[index])  # Agrega una pregunta de esta fuente.
                cursor[source] = index + 1  # Avanza para no repetirla.
                progressed = True  # La ronda sí encontró datos nuevos.
        if not progressed:
            break  # Termina si todas las fuentes agotaron sus preguntas.
    if len(chosen) != per_level:
        raise ValueError(f"Could not sample {per_level} questions from {len(questions)} candidates")  # No devuelve una cuota incompleta.
    return chosen  # La selección alternada reduce el dominio de una sola fuente.


def stratified_sample(
    questions: list[dict[str, Any]],
    per_level: int,
    seed: int = 42,
    levels: list[str] | None = None,
    sources: list[str] | None = None,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """Muestrea la misma cantidad de cada tipo distribuyendo fuentes."""
    levels = levels or LEVELS  # Permite usar los estratos del benchmark o una lista explícita.
    rng = rng or random.Random(seed)  # Inyecta un generador común o crea uno reproducible.
    sources = sources or sorted({str(row["source"]) for row in questions})  # Lista de fuentes disponibles.
    chosen: list[dict[str, Any]] = []  # Une muestras independientes de cada tipo.
    for level in levels:
        pool = [row for row in questions if row["question_type"] == level]  # Candidatas de un único estrato.
        if len(pool) < per_level:
            raise ValueError(f"Not enough questions for {level}: {len(pool)} < {per_level}")  # Detecta cuota imposible.
        chosen.extend(_sample_level(pool, per_level, sources, rng))  # Agrega cuota distribuida por documentos.
    chosen.sort(key=lambda row: (levels.index(row["question_type"]), str(row["id"])))  # Orden canónico del resultado.
    return chosen  # Muestra estratificada, no una selección aleatoria simple.


def sample_sources(records: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    """Enumera las fuentes que aparecen por corpus dentro de una muestra."""
    grouped: dict[str, set[str]] = defaultdict(set)  # Set evita contar repetidamente una fuente.
    for record in records:
        split = "novel" if str(record["id"]).startswith("Novel") else "medical"  # Deriva corpus del prefijo de ID.
        grouped[split].add(str(record["source"]))  # Registra el documento que aportó la pregunta.
    return {split: sorted(sources) for split, sources in grouped.items()}  # Orden estable para el reporte.


def build_sample(
    per_level: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    """Genera una muestra balanceada y devuelve digest y reporte de validación."""
    rng = random.Random(seed)  # Un mismo generador reproduce la secuencia entre corpus y niveles.
    all_records: list[dict[str, Any]] = []  # Acumula Novel y Medical.
    for split in SPLITS:
        questions = load_questions(split, dedupe=True)  # Elimina IDs repetidos antes de seleccionar.
        sources = sorted({str(row["source"]) for row in questions})  # Fuentes desde las que repartir muestras.
        split_records: list[dict[str, Any]] = []  # Resultados temporales del corpus actual.
        for level in LEVELS:
            pool = [row for row in questions if row["question_type"] == level]  # No mezcla tipos de pregunta.
            split_records.extend(_sample_level(pool, per_level, sources, rng))  # Selecciona la cuota para ese tipo.
        split_records.sort(key=lambda row: (LEVELS.index(row["question_type"]), str(row["id"])))  # Ordena para el digest.
        all_records.extend(split_records)  # Añade el corpus completo a la muestra total.
    report = validate_sample(all_records, per_level=per_level)  # Comprueba balance e IDs sin duplicar.
    ids = [str(record["id"]) for record in all_records]  # Orden exacto que alimenta la huella.
    return {
        "seed": seed,  # Semilla que permite volver a generar la selección.
        "per_level": per_level,  # Tamaño de cada tipo por corpus.
        "total": len(all_records),  # Total combinado de los dos corpus.
        "sha256_ids": sha256_text("\n".join(ids)),  # Huella para comprobar identidad y orden.
        "question_ids": ids,  # IDs en el mismo orden que sus registros.
        "records": all_records,  # Registros completos usados por los experimentos.
        "report": report,  # Conteos de balance y fuentes.
    }
