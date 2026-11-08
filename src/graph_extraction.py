"""Extracción de triples por fragmento y construcción de grafos por fuente."""

from __future__ import annotations

import argparse  # Opciones para construir grafos desde la terminal.
import re  # Limpia nombres de fuentes para crear archivos seguros.
import time  # Mide velocidad/progreso de las llamadas al extractor.
from concurrent.futures import ThreadPoolExecutor, as_completed  # Ejecuta varios fragmentos en paralelo.
from pathlib import Path  # Construye rutas a caches, configuraciones y grafos.
from typing import Any, Iterable  # Formas flexibles de respuesta del LLM y triples.

from .chunking import chunk_corpus  # Convierte texto fuente en fragmentos medidos por tokens.
from .data import load_corpus  # Carga el corpus oficial seleccionado.
from .graph import (
    as_triple,
    graph_from_triples,
    graph_metrics,
    is_metadata_triple,
    normalize_entity,
    normalize_relation,
    save_graph,
)
from .io import cache_key, read_json, read_yaml, write_json  # Cachea triples y guarda configuración/resultados.
from .llm import LLMClient  # Cliente para el extractor de triples.
from .paths import CACHE, CONFIGS, GRAPHS, RESULTS  # Rutas del proyecto.
from .prompts import EXTRACTION_SYSTEM, build_extraction_prompt  # Instrucciones que controlan extracción del modelo.

CHUNK_CACHE = CACHE / "chunks"  # Cache de ventanas tokenizadas por documento.
TRIPLE_CACHE = CACHE / "extraction"  # Cache de llamadas de extracción por fragmento.


def safe_name(name: str) -> str:
    """Reemplaza caracteres problemáticos para usar una fuente como nombre de archivo."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)  # Mantiene letras/números y signos seguros.


def chunk_path(split: str, source: str) -> Path:
    """Calcula el JSON de cache de fragmentos para corpus y fuente."""
    return CHUNK_CACHE / split / f"{safe_name(source)}.json"  # Separa corpus para evitar colisiones.


def extraction_path(split: str, source: str, chunk_id: int | str, text: str) -> Path:
    """Calcula la ruta de cache del fragmento, incluyendo huella de su texto."""
    key = cache_key(split, source, str(chunk_id), text)  # Cambio de texto produce otra clave.
    return TRIPLE_CACHE / split / safe_name(source) / f"{key}.json"  # Aísla cada resultado por fuente.


def graph_path(split: str, source: str) -> Path:
    """Devuelve la ruta de salida del grafo construido para una fuente."""
    return GRAPHS / split / f"{safe_name(source)}.json"  # Un JSON por documento fuente.


def get_chunks(
    split: str,
    source: str,
    corpus: dict[str, str],
    chunk_tokens: int = 256,
    chunk_overlap: int = 32,
) -> list[dict[str, Any]]:
    """Carga chunks guardados o divide y cachea el texto si aún no existen."""
    path = chunk_path(split, source)  # Ruta esperada de la cache.
    if path.exists():
        cached = read_json(path)  # Lee archivo existente antes de recalcular el chunking.
        if isinstance(cached, dict) and "chunks" in cached:
            return list(cached["chunks"])  # Devuelve copia de la lista cacheada.
    chunks = chunk_corpus(
        {source: corpus[source]},
        chunk_tokens=chunk_tokens,  # Longitud máxima de cada ventana.
        chunk_overlap=chunk_overlap,  # Tokens compartidos por ventanas consecutivas.
    )[source]
    write_json(path, {"source": source, "chunk_tokens": chunk_tokens, "chunk_overlap": chunk_overlap, "chunks": chunks})  # Guarda parámetros junto al contenido.
    return chunks  # Entrega las ventanas nuevas al extractor.


def validate_triples(items: Any, source: str = "") -> list[tuple[str, str, str]]:
    """Normaliza formas conocidas y elimina triples vacíos, triviales o de metadatos."""
    if isinstance(items, dict):
        items = items.get("triples", items.get("relations", []))  # Acepta respuestas envueltas por una clave superior.
    if not isinstance(items, list):
        return []  # Una respuesta que no sea lista no contiene triples utilizables.
    triples: list[tuple[str, str, str]] = []  # Forma interna homogénea: (head, relation, tail).
    for item in items:
        try:
            head, relation, tail = as_triple(item)  # Admite diccionario o secuencia de tres campos.
        except (TypeError, ValueError):
            continue  # Descarta elementos cuya estructura no representa un triple.
        if not head.strip() or not relation.strip() or not tail.strip():
            continue  # No acepta componentes vacíos.
        if normalize_entity(head) == normalize_entity(tail):
            continue  # Quita auto-relaciones luego de normalizar alias.
        if is_metadata_triple(head, relation, tail, source=source):
            continue  # Evita que encabezados/créditos contaminen el grafo del documento.
        triples.append((head.strip(), relation.strip(), tail.strip()))  # Conserva el texto limpio del modelo.
    return triples  # Devuelve triples filtrados para deduplicación y construcción.


def dedupe_triples(triples: Iterable[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """Retira triples repetidos comparando extremos y relación normalizados."""
    seen: set[tuple[str, str, str]] = set()  # Claves canónicas ya emitidas.
    unique: list[tuple[str, str, str]] = []  # Representaciones originales que se conservarán.
    for head, relation, tail in triples:
        key = (normalize_entity(head), normalize_relation(relation), normalize_entity(tail))  # Ignora diferencias de mayúsculas/puntuación.
        if key in seen:
            continue  # Solo guarda una copia semánticamente equivalente.
        seen.add(key)  # Marca el triple antes de seguir recorriendo.
        unique.append((head, relation, tail))  # Conserva el primer texto original encontrado.
    return unique  # Lista de triples únicos.


def extract_chunk(
    split: str,
    source: str,
    chunk: dict[str, Any],
    client: LLMClient,
    overwrite: bool = False,
) -> list[tuple[str, str, str]]:
    """Usa triples cacheados o llama al LLM y guarda la extracción validada."""
    path = extraction_path(split, source, chunk["chunk_id"], str(chunk["text"]))  # Cache identifica el texto exacto del fragmento.
    if path.exists() and not overwrite:
        cached = read_json(path)  # Evita costo/latencia si la extracción ya está hecha.
        if isinstance(cached, dict) and "triples" in cached:
            return validate_triples(cached["triples"], source=source)  # Revalida para evitar usar cache malformada.
    prompt = build_extraction_prompt(source, chunk["chunk_id"], str(chunk["text"]))  # Incluye fuente, ID y texto.
    items = client.complete_json(EXTRACTION_SYSTEM, prompt)  # Pide respuesta estructurada al modelo.
    triples = validate_triples(items, source=source)  # Limpia forma, vacíos, auto-relaciones y metadatos.
    write_json(
        path,
        {
            "split": split,  # Corpus del que proviene la extracción.
            "source": source,  # Documento concreto de origen.
            "chunk_id": chunk["chunk_id"],  # Ventana que el modelo leyó.
            "triples": [list(triple) for triple in triples],  # JSON serializable de las relaciones aceptadas.
        },
    )
    return triples  # Devuelve la misma información que quedó guardada.


def extract_source(
    split: str,
    source: str,
    chunks: list[dict[str, Any]],
    client: LLMClient | None,
    workers: int = 5,
    overwrite: bool = False,
) -> list[tuple[str, str, str]]:
    """Extrae triples de chunks en paralelo, informa progreso y deduplica al final."""
    if client is None:
        return []  # Dry-run sin cliente solo prueba chunking; no inventa extracciones.
    triples: list[tuple[str, str, str]] = []  # Resultados parciales de todas las ventanas.
    completed = 0  # Cantidad de futuros terminados.
    started = time.time()  # Referencia para calcular velocidad.
    interval = max(1, len(chunks) // 10)  # Informa aproximadamente cada 10% del trabajo.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Envía una tarea por chunk; el diccionario conserva el chunk asociado a cada futuro.
        futures = {
            executor.submit(extract_chunk, split, source, chunk, client, overwrite): chunk
            for chunk in chunks
        }
        for future in as_completed(futures):
            triples.extend(future.result())  # Si la tarea falló, future.result propaga la excepción.
            completed += 1  # Cuenta una extracción completada.
            if completed % interval == 0 or completed == len(chunks):
                elapsed = time.time() - started  # Tiempo desde que empezó el grupo concurrente.
                rate = completed / elapsed if elapsed else 0.0  # Evita división por cero si fue instantáneo.
                print(
                    f"[{split}/{source}] {completed}/{len(chunks)} chunks "
                    f"| {len(triples)} triples | {rate:.2f} chunks/s",
                    flush=True,
                )
    return dedupe_triples(triples)  # Deduplica cuando ya llegaron todos los chunks.


def build_source_graph(
    split: str,
    source: str,
    client: LLMClient | None,
    chunk_tokens: int = 256,
    chunk_overlap: int = 32,
    workers: int = 5,
    overwrite: bool = False,
    limit_chunks: int | None = None,
) -> dict[str, Any]:
    """Construye el grafo de una fuente y lo guarda si se ejecuta extracción real."""
    corpus = load_corpus(split)  # Carga los textos disponibles para validar source.
    if source not in corpus:
        raise KeyError(f"Unknown source {source!r} for split {split}")  # Falla si el nombre no corresponde al corpus.
    chunks = get_chunks(
        split,
        source,
        corpus,
        chunk_tokens=chunk_tokens,  # Ventana que recibe el extractor.
        chunk_overlap=chunk_overlap,  # Solapamiento para no cortar evidencia entre ventanas.
    )
    if limit_chunks is not None:
        chunks = chunks[:limit_chunks]  # Limita una prueba rápida sin alterar los archivos fuente.
    triples = extract_source(
        split,
        source,
        chunks,
        client,
        workers=workers,  # Paralelismo dentro de la fuente.
        overwrite=overwrite,  # Permite ignorar entradas de extracción cacheadas.
    )
    graph = graph_from_triples(triples)  # Construye el grafo a partir de triples validados.
    metrics = graph_metrics(graph)  # Calcula un resumen estructural del resultado.
    dry_run = client is None  # Sin cliente no se invoca el LLM ni se crea un grafo persistente.
    path = None  # En dry-run no hay ruta de grafo final.
    if not dry_run:
        path = str(graph_path(split, source))  # Determina el JSON de esta fuente.
        save_graph(path, graph)  # Persiste nodos y relaciones.
    return {
        "split": split,  # Corpus que se procesó.
        "source": source,  # Documento concreto.
        "chunks": len(chunks),  # Fragmentos considerados.
        "triples": len(triples),  # Relaciones finales sin duplicados.
        "metrics": metrics,  # Tamaño y conectividad del grafo.
        "path": path,  # Ruta guardada, o None cuando es una prueba seca.
        "dry_run": dry_run,  # Hace explícito si hubo persistencia/llamadas.
    }


def build_split_graphs(
    split: str,
    sources: list[str] | None = None,
    client: LLMClient | None = None,
    chunk_tokens: int = 256,
    chunk_overlap: int = 32,
    workers: int = 5,
    overwrite: bool = False,
    limit_chunks: int | None = None,
) -> list[dict[str, Any]]:
    """Recorre fuentes de un corpus y agrega un resumen por grafo."""
    corpus = load_corpus(split)  # Permite listar automáticamente las fuentes disponibles.
    selected = sources or sorted(corpus)  # Usa selección explícita o todas en orden estable.
    summaries = []  # Acumula reportes por fuente.
    for index, source in enumerate(selected, start=1):
        print(f"[{split}] source {index}/{len(selected)}: {source}", flush=True)  # Muestra progreso entre documentos.
        summary = build_source_graph(
            split,
            source,
            client,
            chunk_tokens=chunk_tokens,  # Reenvía configuración de chunking.
            chunk_overlap=chunk_overlap,
            workers=workers,  # Reenvía cantidad de tareas paralelas.
            overwrite=overwrite,  # Reenvía política de cache.
            limit_chunks=limit_chunks,  # Reenvía límite de prueba.
        )
        metrics = summary["metrics"]  # Facilita imprimir conteos clave.
        print(
            f"[{split}] done {source}: chunks={summary['chunks']} "
            f"triples={summary['triples']} nodes={metrics['node_count']} "
            f"edges={metrics['edge_count']}",
            flush=True,
        )
        summaries.append(summary)  # Conserva el detalle de esta fuente para el manifiesto.
    return summaries  # Lista de resúmenes para uno o varios corpus.


def main() -> None:
    """Procesa opciones CLI y lanza generación para las fuentes seleccionadas."""
    parser = argparse.ArgumentParser(description="Build per-source knowledge graphs")  # Configura ayuda y errores de uso.
    parser.add_argument("--split", choices=["novel", "medical", "all"], default="all")  # Corpus objetivo.
    parser.add_argument("--sources", default=None, help="Comma-separated source names")  # Subconjunto opcional de documentos.
    parser.add_argument("--chunk-tokens", type=int, default=256)  # Tamaño de fragmento.
    parser.add_argument("--chunk-overlap", type=int, default=32)  # Tokens repetidos entre fragmentos.
    parser.add_argument("--workers", type=int, default=5)  # Llamadas de extracción simultáneas.
    parser.add_argument("--limit-chunks", type=int, default=None)  # Límite para pruebas reducidas.
    parser.add_argument("--overwrite", action="store_true")  # Fuerza a recalcular cache de triples.
    parser.add_argument("--dry-run", action="store_true", help="Chunk only, do not call the model")  # Omite llamadas LLM.
    parser.add_argument("--config", default=str(CONFIGS / "graph.yaml"))  # Parámetros del extractor.
    args = parser.parse_args()  # Lee los valores suministrados desde la terminal.

    config = read_yaml(args.config) if Path(args.config).exists() else {}  # Permite ejecutar sin archivo de configuración.
    extractor = config.get("extractor", {})  # Aísla las opciones de la sección del extractor.
    client = None  # None activa el modo de chunks sin llamadas al modelo.
    if not args.dry_run:
        client = LLMClient(
            model=extractor.get("model", "gpt-4o-mini"),  # Modelo definido o default.
            temperature=float(extractor.get("temperature", 0.0)),  # Convierte YAML a número.
            max_tokens=int(extractor.get("max_output_tokens", 1024)),  # Límite de respuesta para triple extraction.
        )
    sources = [source.strip() for source in args.sources.split(",")] if args.sources else None  # Separa lista de fuentes opcional.
    splits = ["novel", "medical"] if args.split == "all" else [args.split]  # Expande all a ambos corpus.
    summaries: list[dict[str, Any]] = []  # Resultado global para escribir en el manifiesto.
    for split in splits:
        summaries.extend(
            build_split_graphs(
                split,
                sources=sources,  # Mismo filtro opcional para cada corpus.
                client=client,  # None significa dry-run.
                chunk_tokens=args.chunk_tokens,  # Configuración de ventanas.
                chunk_overlap=args.chunk_overlap,
                workers=args.workers,  # Paralelismo de extracción.
                overwrite=args.overwrite,  # Política de cache de triples.
                limit_chunks=args.limit_chunks,  # Límite de depuración.
            )
        )
    write_json(RESULTS / "graph_build.json", summaries)  # Guarda el resumen auditable de todas las fuentes.
    print(f"Processed {len(summaries)} source graphs")  # Indica cuántos documentos se procesaron.


if __name__ == "__main__":
    main()
