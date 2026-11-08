"""Definición de condiciones y construcción de contextos experimentales."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx  # Tipo de grafo usado por recursos e índices.

from .budget import count_tokens, pack_contexts  # Control de longitud y conteo de tokens.
from .chunking import chunk_corpus  # Prepara textos en pasajes recuperables.
from .data import load_corpus  # Lee los textos oficiales por corpus.
from .graph import load_graph, rewire_edges  # Carga grafos y crea variantes rewired.
from .graph_retrieval import GraphContext, GraphIndex  # Selección de subgrafos relevantes.
from .retrieval import TfidfRetriever  # Baseline léxico basado en TF-IDF.

Q1_CONDITIONS = ["rag", "graph_content", "graph_full", "graph_rewired"]  # Ablaciones y baseline de Q1.
Q2_CONDITIONS = ["rag", "graph_full", "graph_greedy", "graph_adaptive"]  # Métodos de selección comparados en Q2.
ALL_CONDITIONS = [  # Nombres permitidos por el generador de contexto.
    "rag",
    "graph_content",
    "graph_full",
    "graph_rewired",
    "graph_greedy",
    "graph_adaptive",
    "graph_random_edges",
    "graph_random_prune",
    "graph_centrality_prune",
]


@dataclass
class ExperimentResources:
    """Share corpus retrieval resources and lazily cached per-graph indexes."""

    retriever: TfidfRetriever  # Índice de pasajes para la condición RAG.
    graphs: dict[str, nx.Graph]  # Grafo original por fuente/corpus.
    rag_top_k: int = 10  # Máximo de fragmentos candidatos recuperados por RAG.
    graph_hops: int = 2  # Profundidad usada por las condiciones de grafo estándar.
    seed_k: int = 10  # Cantidad de entidades semilla para explorar un grafo.
    encoding_name: str = "cl100k_base"  # Tokenizador con el que se miden presupuestos.
    rewire_seed: int = 42  # Semilla que fija la aleatoriedad de rewiring.
    aux_graphs: dict[str, dict[str, nx.Graph]] = field(default_factory=dict)  # Variantes precalculadas por condición/fuente.
    _indexes: dict[tuple[str, str], GraphIndex] = field(default_factory=dict)  # Cache de índices ya construidos.

    def graph_index(self, source: str, condition: str) -> GraphIndex:
        """Obtiene el índice correspondiente y conserva su copia para futuros usos."""
        key = (source, condition)  # Índice distinto por fuente y transformación.
        if key in self._indexes:
            return self._indexes[key]  # Evita reindexar el mismo grafo en cada pregunta.
        if source not in self.graphs:
            raise KeyError(f"No graph available for source {source!r}")  # Informa fuente sin grafo.
        graph = self.graphs[source]  # Parte del grafo original del documento.
        if condition == "graph_rewired" or condition == "graph_random_edges":
            graph = rewire_edges(graph, seed=self.rewire_seed)  # Cambia parejas manteniendo grados cuando es posible.
        elif condition in self.aux_graphs and source in self.aux_graphs[condition]:
            graph = self.aux_graphs[condition][source]  # Usa una variante ya calculada si se proporcionó.
        index = GraphIndex(graph, seed_k=self.seed_k)  # Construye índice de nodos/aristas consultable.
        self._indexes[key] = index  # Guarda el índice para la siguiente pregunta de esa fuente.
        return index  # Devuelve el índice de la condición elegida.


def _rag_contexts(
    question: str,
    source: str,
    budget: int,
    resources: ExperimentResources,
) -> tuple[list[str], dict[str, Any]]:
    """Recupera pasajes de texto del mismo documento y aplica el presupuesto."""
    results = resources.retriever.search(question, top_k=resources.rag_top_k, source=source)  # Recupera pasajes más similares.
    texts = [result.text for result in results]  # Descarta score/metadata para formar el prompt.
    contexts, used = pack_contexts(texts, max_tokens=budget, encoding_name=resources.encoding_name)  # Recorta al límite solicitado.
    metadata = {
        "strategy": "rag",  # Identifica el método aplicado.
        "retrieved": len(results),  # Cantidad de pasajes candidatos antes del presupuesto.
        "selected": len(contexts),  # Cantidad de bloques que sí se enviaron.
        "scores": [round(result.score, 6) for result in results],  # Relevancia TF-IDF para auditoría.
        "chunk_ids": [result.chunk_id for result in results],  # IDs de pasajes recuperados.
        "context_tokens": used,  # Tokens medidos del texto seleccionado.
    }
    return contexts, metadata  # Contexto y trazabilidad para la predicción.


def _graph_context(
    condition: str,
    question: str,
    source: str,
    budget: int,
    resources: ExperimentResources,
) -> tuple[list[str], dict[str, Any]]:
    """Construye contexto de grafo usando el algoritmo asociado a la condición."""
    index = resources.graph_index(source, condition)  # Recupera índice original o transformado.
    if condition == "graph_content":
        result = index.content_context(question, budget, hops=resources.graph_hops, seed_k=resources.seed_k)  # Solo nombres de nodos.
    elif condition == "graph_greedy":
        result = index.greedy_context(question, budget, hops=resources.graph_hops, seed_k=resources.seed_k)  # Maximiza cobertura por token.
    elif condition == "graph_adaptive":
        result = index.adaptive_context(question, budget, max_hops=max(2, resources.graph_hops + 1), seed_k=resources.seed_k)  # Expande alcance con atenuación.
    else:
        result = index.full_context(question, budget, hops=resources.graph_hops, seed_k=resources.seed_k)  # Selecciona aristas del subgrafo.
    metadata = result.to_metadata()  # Extrae estrategia, tamaño y elementos seleccionados.
    metadata["context_tokens"] = sum(
        count_tokens(line, encoding_name=resources.encoding_name) for line in result.contexts  # Mide cada línea que formará el prompt.
    )
    metadata["edge_count"] = len(result.edges)  # Registra relaciones seleccionadas.
    metadata["node_count"] = len(result.nodes)  # Registra entidades seleccionadas.
    return result.contexts, metadata  # Devuelve texto y datos de auditoría.


def build_contexts(
    record: dict[str, Any],
    condition: str,
    budget: int,
    resources: ExperimentResources,
) -> tuple[list[str], dict[str, Any]]:
    """Genera contexto y metadatos de una pregunta en una condición dada."""
    if condition not in ALL_CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")  # No acepta nombres fuera del protocolo.
    source = str(record["source"])  # Selecciona el grafo/texto del documento de la pregunta.
    question = str(record["question"])  # Normaliza el prompt a cadena.
    if condition == "rag":
        contexts, metadata = _rag_contexts(question, source, budget, resources)  # El baseline usa fragmentos textuales.
    else:
        contexts, metadata = _graph_context(condition, question, source, budget, resources)  # El resto usa el índice de grafo.
    metadata.update({"condition": condition, "budget": budget, "source": source})  # Anexa claves compartidas por condiciones.
    return contexts, metadata  # Mantiene mismo formato entre RAG y grafos.


def load_resources(
    split: str,
    chunk_tokens: int = 256,
    chunk_overlap: int = 32,
    graph_dir: str | Path | None = None,
) -> ExperimentResources:
    """Prepara pasajes TF-IDF y grafos de un corpus antes de ejecutar condiciones."""
    corpus = load_corpus(split)  # Lee textos del corpus solicitado.
    chunks: list[dict[str, Any]] = []  # Acumula chunks de todas las fuentes.
    for source, text in corpus.items():
        source_chunks = chunk_corpus(
            {source: text},
            chunk_tokens=chunk_tokens,  # Tamaño de ventana en tokens.
            chunk_overlap=chunk_overlap,  # Contexto compartido entre ventanas vecinas.
        )[source]
        for chunk in source_chunks:
            chunk["source"] = source  # Permite filtrar RAG al documento de la pregunta.
            chunks.append(chunk)  # Añade cada fragmento al índice común.
    retriever = TfidfRetriever().fit(chunks)  # Aprende TF-IDF para ese corpus.
    root = Path(graph_dir) if graph_dir is not None else None  # Convierte el directorio si el usuario lo indicó.
    graphs: dict[str, nx.Graph] = {}  # Mapa de grafo por documento fuente.
    if root is not None and root.exists():
        for path in sorted(root.glob("*.json")):
            source = path.stem  # El nombre del archivo identifica la fuente.
            try:
                graphs[source] = load_graph(path)  # Reconstruye la estructura de NetworkX.
            except Exception:
                continue  # Conserva el comportamiento previo de saltar grafos ilegibles.
    return ExperimentResources(retriever=retriever, graphs=graphs)  # Comparte recursos entre preguntas.
