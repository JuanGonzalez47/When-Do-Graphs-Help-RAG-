"""Índices y métodos para seleccionar contexto desde grafos de conocimiento."""

from __future__ import annotations

import re  # Tokeniza consultas/predicados para estimar cobertura textual.
from collections import deque  # Recorrido en anchura para vecindarios del grafo.
from dataclasses import dataclass, field  # Estructuras compactas con listas independientes.
from typing import Any, Iterable  # Tipos para grafos y líneas seleccionadas.

import networkx as nx  # Representación del grafo y cálculo de grados.

from .budget import count_tokens  # Cuenta tokens antes de añadir líneas al contexto.
from .graph import to_simple_graph  # Uniforma grafos dirigidos/no dirigidos para recuperar.
from .retrieval import TfidfRetriever  # Índice TF-IDF de nombres y texto de aristas.

STOPWORDS = {  # Palabras funcionales que rara vez ayudan a localizar entidades/relaciones.
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def query_terms(text: str) -> set[str]:
    """Extrae términos alfanuméricos de la consulta y elimina stopwords."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())  # Divide texto en letras/números en minúscula.
    return {token for token in tokens if token not in STOPWORDS and len(token) > 1}  # Quita términos demasiado comunes/cortos.


def edge_text(graph: nx.Graph, head: str, tail: str) -> str:
    """Convierte una arista en texto que el generador pueda leer."""
    relations = graph[head][tail].get("relations", [])  # Recupera todas las etiquetas asociadas a esa pareja.
    if relations:
        return f"{head} -- {' ; '.join(relations)} -- {tail}"  # Representa explícitamente nodo, relación y nodo.
    return f"{head} -- {tail}"  # Fallback para aristas sin etiquetas.


@dataclass
class GraphContext:
    """Hold the selected context lines, their graph elements, and retrieval metadata."""

    contexts: list[str]  # Líneas de texto que se entregan al modelo.
    edges: list[tuple[str, str]] = field(default_factory=list)  # Parejas de nodos representadas.
    nodes: list[str] = field(default_factory=list)  # Entidades presentes en las aristas elegidas.
    metadata: dict[str, Any] = field(default_factory=dict)  # Estrategia, hops, cobertura y conteos.

    def to_metadata(self) -> dict[str, Any]:
        """Copia metadatos de búsqueda y añade tamaños calculados."""
        data = dict(self.metadata)  # Evita modificar el diccionario interno del objeto.
        data.update(
            {
                "contexts": len(self.contexts),  # Cantidad de bloques serializados.
                "selected_edges": len(self.edges),  # Cantidad de aristas utilizadas.
                "selected_nodes": len(self.nodes),  # Cantidad de entidades utilizadas.
            }
        )
        return data  # Metadatos listos para adjuntar a la predicción.


class GraphIndex:
    """Indexa nombres y relaciones para recuperar subgrafos relevantes a una pregunta."""

    def __init__(
        self,
        graph: nx.Graph,
        ngram_range: tuple[int, int] = (1, 2),
        seed_k: int = 10,
    ) -> None:
        self.graph = to_simple_graph(graph)  # Copia el grafo a una forma coherente para los algoritmos.
        self.seed_k = seed_k  # Máximo de nodos semilla iniciales.
        nodes = sorted(self.graph.nodes())  # Orden estable para alinear cada vector con su ID.
        edges = sorted((min(head, tail), max(head, tail)) for head, tail in self.graph.edges())  # Canoniza extremos no dirigidos.
        self.nodes = nodes  # Guarda el orden usado por el índice de nodos.
        self.edges = edges  # Guarda el orden usado por el índice de aristas.
        self.node_retriever = TfidfRetriever(ngram_range=ngram_range)  # Comparará consulta con nombres de entidad.
        self.edge_retriever = TfidfRetriever(ngram_range=ngram_range)  # Comparará consulta con triples serializados.
        if nodes:
            self.node_retriever.fit(
                [{"chunk_id": node, "source": "node", "text": node} for node in nodes]  # Un documento TF-IDF por nodo.
            )
        if edges:
            self.edge_retriever.fit(
                [
                    {
                        "chunk_id": f"{head}\x1f{tail}",  # ID estable de la pareja de extremos.
                        "source": "edge",  # Distingue documentos de arista de documentos de nodo.
                        "text": edge_text(self.graph, head, tail),  # Texto con nombres y relaciones.
                    }
                    for head, tail in edges
                ]
            )

    def _node_scores(self, query: str) -> dict[str, float]:
        if not self.nodes:
            return {}  # No puede consultarse un índice sin nodos.
        scores = self.node_retriever.scores(query)  # Una relevancia por nodo, en orden self.nodes.
        return {node: float(scores[index]) for index, node in enumerate(self.nodes)}  # Convierte vector en mapa legible.

    def _edge_scores(self, query: str) -> dict[tuple[str, str], float]:
        if not self.edges:
            return {}  # Un grafo sin aristas no tiene puntuaciones de relación.
        scores = self.edge_retriever.scores(query)  # Similaridad TF-IDF para cada texto de arista.
        return {edge: float(scores[index]) for index, edge in enumerate(self.edges)}  # Asocia similitud a sus extremos.

    def _seed_nodes(self, query: str, seed_k: int | None = None) -> list[str]:
        """Busca entidades relevantes iniciales; usa grado como fallback léxico."""
        seed_k = self.seed_k if seed_k is None else seed_k  # Permite reemplazar la cuota por consulta.
        scores = self._node_scores(query)  # Puntúa nombres de nodos contra la pregunta.
        if not scores:
            return []  # No hay nodos indexados.
        ranked = sorted(scores, key=lambda node: scores[node], reverse=True)  # Ordena de similitud mayor a menor.
        selected = [node for node in ranked[:seed_k] if scores[node] > 0]  # Excluye coincidencias de puntuación cero.
        if selected:
            return selected  # Usa los términos de la pregunta cuando coinciden con nombres.
        return sorted(self.graph.nodes(), key=lambda node: self.graph.degree(node), reverse=True)[:seed_k]  # Si no hay match, parte de nodos conectados.

    def _edge_anchors(self, query: str, limit: int) -> list[tuple[str, str]]:
        """Selecciona aristas de coincidencia léxica positiva como anclas."""
        edge_scores = self._edge_scores(query)  # Puntuación de cada relación textual.
        ranked = sorted(
            (edge for edge, score in edge_scores.items() if score > 0),
            key=lambda edge: edge_scores[edge],
            reverse=True,  # Prioriza las relaciones más similares a la pregunta.
        )
        return ranked[:limit]  # Limita el número de aristas ancla.

    def _seeds_with_anchors(self, query: str, seed_k: int | None = None) -> tuple[list[str], list[tuple[str, str]]]:
        """Combina los nodos semilla con los extremos de aristas coincidentes."""
        seed_k = self.seed_k if seed_k is None else seed_k  # Usa cuota predeterminada o valor recibido.
        node_seeds = self._seed_nodes(query, seed_k=seed_k)  # Recupera nombres directamente relacionados.
        anchors = self._edge_anchors(query, limit=max(seed_k, 20))  # Añade un conjunto mínimo de aristas candidatas.
        anchor_nodes = {node for edge in anchors for node in edge}  # Extrae extremos de esas aristas.
        return list(dict.fromkeys([*node_seeds, *sorted(anchor_nodes)])), anchors  # Elimina duplicados sin cambiar orden.

    def _neighborhood(self, seeds: Iterable[str], hops: int) -> set[str]:
        """Expande desde semillas hasta una profundidad máxima en saltos."""
        visited = set(seeds)  # Incluye las semillas aunque hops sea cero.
        frontier = deque((seed, 0) for seed in seeds)  # Cola de nodo y distancia desde semilla.
        while frontier:
            node, depth = frontier.popleft()  # Procesa primero el nodo más cercano.
            if depth >= hops:
                continue  # No explora vecinos más allá del máximo autorizado.
            for neighbor in self.graph.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)  # Marca para evitar volver a insertar el nodo.
                    frontier.append((neighbor, depth + 1))  # Agenda el vecino para el siguiente nivel.
        return visited  # Conjunto de nodos dentro del radio de búsqueda.

    def _candidate_edges(self, nodes: set[str]) -> list[tuple[str, str]]:
        """Filtra aristas cuyos dos extremos aparecen en el vecindario."""
        return [edge for edge in self.edges if edge[0] in nodes and edge[1] in nodes]  # No cruza fuera del subgrafo.

    def _rank_edges(
        self,
        edges: Iterable[tuple[str, str]],
        edge_scores: dict[tuple[str, str], float],
        node_scores: dict[str, float],
    ) -> list[tuple[str, str]]:
        """Ordena aristas combinando relevancia de relación, nodos y grado."""
        max_node = max(node_scores.values(), default=1.0) or 1.0  # Evita división por cero al normalizar.
        return sorted(
            edges,
            key=lambda edge: (
                edge_scores.get(edge, 0.0)  # Coincidencia directa del texto de arista.
                + 0.5 * (node_scores.get(edge[0], 0.0) + node_scores.get(edge[1], 0.0)) / max_node
                + 0.001 * (self.graph.degree(edge[0]) + self.graph.degree(edge[1]))  # Preferencia débil por extremos conectados.
            ),
            reverse=True,  # Los valores mayores se seleccionan primero.
        )

    def _pack_lines(self, lines: Iterable[tuple[Any, str]], budget: int) -> tuple[list[str], list[Any]]:
        """Agrega líneas completas mientras no excedan el presupuesto de tokens."""
        selected_lines: list[str] = []  # Texto que formará el contexto.
        selected_edges: list[Any] = []  # IDs de nodos/aristas asociados al mismo texto.
        used = 0  # Tokens acumulados en líneas completas.
        for edge, line in lines:
            cost = count_tokens(line)  # Mide una línea con el tokenizador predeterminado.
            if used + cost > budget:
                continue  # Omite una línea que no cabe; aún podría caber una siguiente más corta.
            selected_lines.append(line)  # Añade la representación textual.
            selected_edges.append(edge)  # Mantiene alineado el elemento estructural.
            used += cost  # Actualiza el presupuesto consumido.
        return selected_lines, selected_edges  # Devuelve texto y elementos con el mismo orden.

    def full_context(
        self,
        query: str,
        budget: int,
        hops: int = 2,
        seed_k: int | None = None,
    ) -> GraphContext:
        """Ordena aristas del vecindario y las incorpora respetando el presupuesto."""
        if not self.edges:
            return GraphContext([], metadata={"strategy": "full", "graph_nodes": len(self.nodes)})  # Grafo sin contexto relacional.
        seeds, anchors = self._seeds_with_anchors(query, seed_k)  # Nodos y aristas conectados léxicamente a la consulta.
        edge_scores = self._edge_scores(query)  # Similitud de todos los textos de arista con la pregunta.
        neighborhood = self._neighborhood(seeds, hops)  # Limita búsqueda al número de saltos configurado.
        candidates = self._candidate_edges(neighborhood)  # Solo usa aristas internas al subgrafo.
        node_scores = self._node_scores(query)  # Relevancia de extremos como apoyo al ranking.
        ranked = self._rank_edges(candidates, edge_scores, node_scores)  # Ordena aristas por combinación de señales.
        lines = [(edge, edge_text(self.graph, *edge)) for edge in ranked]  # Convierte aristas en triples legibles.
        selected_lines, selected_edges = self._pack_lines(lines, budget)  # Toma líneas completas que caben.
        selected_nodes = sorted({node for edge in selected_edges for node in edge})  # Nodos presentes en esas líneas.
        return GraphContext(
            contexts=selected_lines,  # Texto para el prompt del LLM.
            edges=selected_edges,  # Aristas usadas para auditar el subgrafo.
            nodes=selected_nodes,  # Extremos usados por las aristas.
            metadata={
                "strategy": "full",  # Identifica recuperación de aristas con predicados.
                "hops": hops,  # Profundidad máxima desde las semillas.
                "seeds": len(seeds),  # Cantidad de semillas combinadas.
                "edge_anchors": len(anchors),  # Coincidencias léxicas directas con aristas.
                "neighborhood": len(neighborhood),  # Nodos dentro del radio consultado.
                "candidates": len(candidates),  # Aristas candidatas antes del presupuesto.
                "graph_nodes": len(self.nodes),  # Tamaño del grafo fuente.
                "graph_edges": len(self.edges),  # Aristas del grafo fuente.
            },
        )

    def content_context(
        self,
        query: str,
        budget: int,
        hops: int = 2,
        seed_k: int | None = None,
    ) -> GraphContext:
        """Devuelve nombres de entidades sin las etiquetas de relación."""
        full = self.full_context(query, budget, hops=hops, seed_k=seed_k)  # Reutiliza selección de aristas para aislar contenido.
        if not full.edges:
            return GraphContext([], metadata={"strategy": "content", "graph_nodes": len(self.nodes)})  # Ninguna arista elegida.
        nodes = sorted({node for edge in full.edges for node in edge})  # Recupera extremos únicos de las aristas seleccionadas.
        lines = [(node, node) for node in nodes]  # Cada nodo se trata como una línea independiente.
        selected_lines, selected_nodes = self._pack_lines(lines, budget)  # Aplica el mismo límite de tokens al contenido.
        return GraphContext(
            contexts=selected_lines,  # Nombres de nodos enviados al generador.
            edges=[],  # No se presentan relaciones en esta condición.
            nodes=list(selected_nodes),  # Nodos que cupieron en el presupuesto.
            metadata={**full.metadata, "strategy": "content"},  # Mantiene trazas y marca la ablación.
        )

    def greedy_context(
        self,
        query: str,
        budget: int,
        hops: int = 2,
        seed_k: int | None = None,
    ) -> GraphContext:
        """Selecciona aristas con mejor cobertura nueva de términos por token."""
        if not self.edges:
            return GraphContext([], metadata={"strategy": "greedy", "graph_nodes": len(self.nodes)})  # Grafo sin relaciones.
        terms = query_terms(query)  # Palabras informativas que se intentará cubrir.
        seeds, anchors = self._seeds_with_anchors(query, seed_k)  # Punto de inicio con señales de nodos y aristas.
        neighborhood = self._neighborhood(seeds, hops)  # Restringe candidatos a los saltos permitidos.
        candidates = self._candidate_edges(neighborhood)  # Aristas del vecindario.
        edge_scores = self._edge_scores(query)  # Relevancia léxica de las aristas.
        remaining = list(candidates)  # Aristas aún no seleccionadas en el ciclo greedy.
        covered: set[str] = set()  # Términos de la consulta ya representados en el contexto.
        selected_lines: list[str] = []  # Líneas seleccionadas hasta el momento.
        selected_edges: list[tuple[str, str]] = []  # Aristas que corresponden a esas líneas.
        used = 0  # Tokens acumulados en la selección.
        while remaining:
            best_edge = None  # Mejor candidata de esta iteración.
            best_line = ""  # Representación textual de esa candidata.
            best_score = -1.0  # Puntuación inferior a cualquier candidata válida.
            for edge in remaining:
                line = edge_text(self.graph, *edge)  # Serializa nodos y predicados.
                line_terms = query_terms(line)  # Términos que contiene esta arista.
                gain = len((line_terms & terms) - covered)  # Cuenta términos de consulta aún no cubiertos.
                cost = count_tokens(line)  # Costo tokenizado de añadir la línea.
                if used + cost > budget:
                    continue  # Candidata demasiado costosa no puede elegirse en esta iteración.
                degree = self.graph.degree(edge[0]) + self.graph.degree(edge[1])  # Señal débil de conectividad de extremos.
                score = (gain + 0.01 * degree + 0.001 * edge_scores.get(edge, 0.0)) / (cost + 1)  # Beneficio aproximado por token.
                if score > best_score:
                    best_edge = edge  # Guarda el ID de la mejor candidata hasta ahora.
                    best_line = line  # Guarda su texto para no volver a serializar.
                    best_score = score  # Actualiza máximo de la ronda.
            if best_edge is None:
                break  # Ninguna arista restante cabe en el presupuesto.
            selected_edges.append(best_edge)  # Registra el elemento estructural elegido.
            selected_lines.append(best_line)  # Añade su texto al contexto.
            used += count_tokens(best_line)  # Actualiza consumo real.
            covered.update(query_terms(best_line) & terms)  # Marca palabras que ya están representadas.
            remaining.remove(best_edge)  # Evita elegir la misma arista de nuevo.
            if used >= budget:
                break  # Termina al llenar el límite.
        selected_nodes = sorted({node for edge in selected_edges for node in edge})  # Entidades de aristas elegidas.
        return GraphContext(
            contexts=selected_lines,  # Texto que recibe el LLM.
            edges=selected_edges,  # Aristas seleccionadas.
            nodes=selected_nodes,  # Nodos seleccionados.
            metadata={
                "strategy": "greedy",  # Identifica la selección incremental.
                "hops": hops,  # Radio de candidatos.
                "seeds": len(seeds),  # Semillas de búsqueda.
                "edge_anchors": len(anchors),  # Aristas ancla encontradas.
                "neighborhood": len(neighborhood),  # Nodos inspeccionados.
                "candidates": len(candidates),  # Aristas antes de la selección.
                "query_coverage": len(covered) / (len(terms) or 1),  # Fracción de términos cubierta.
                "graph_nodes": len(self.nodes),  # Nodos del grafo original.
                "graph_edges": len(self.edges),  # Aristas del grafo original.
            },
        )

    def adaptive_context(
        self,
        query: str,
        budget: int,
        max_hops: int = 3,
        decay: float = 0.5,
        seed_k: int | None = None,
    ) -> GraphContext:
        """Expande puntuaciones por saltos y conserva el nivel con mejor cobertura."""
        if not self.edges:
            return GraphContext([], metadata={"strategy": "adaptive", "graph_nodes": len(self.nodes)})  # No hay contexto relacional posible.
        terms = query_terms(query)  # Palabras cuya presencia mide cobertura.
        base_scores = self._node_scores(query)  # Importancia léxica inicial por nodo.
        seeds, anchors = self._seeds_with_anchors(query, seed_k)  # Semillas y extremos de aristas coincidentes.
        edge_scores = self._edge_scores(query)  # Relevancia textual de las relaciones.
        propagation = {node: base_scores.get(node, 0.0) for node in seeds}  # Estado de activación en el salto actual.
        best = GraphContext([], metadata={"strategy": "adaptive", "graph_nodes": len(self.nodes)})  # Mejor resultado visto.
        best_rank = (-1.0, -1)  # Inicializa una pareja cobertura/número de aristas inferior a cualquier resultado.
        for hop in range(1, max_hops + 1):
            updated = dict(propagation)  # Los nodos mantienen al menos su puntuación previa.
            for node, score in propagation.items():
                for neighbor in self.graph.neighbors(node):
                    candidate = score * decay  # Atenúa la señal a medida que se aleja de la semilla.
                    if candidate > updated.get(neighbor, 0.0):
                        updated[neighbor] = candidate  # Guarda la mayor activación recibida por ese nodo.
            propagation = updated  # Avanza al estado del salto actual.
            active = {node for node, score in propagation.items() if score > 0}  # Nodos alcanzados con señal positiva.
            candidates = self._candidate_edges(active)  # Aristas entre nodos activados.
            if not candidates:
                continue  # Prueba el siguiente salto si no hay relaciones entre nodos activos.
            ranked = self._rank_edges(candidates, edge_scores, propagation)  # Ordena con puntuaciones propagadas.
            lines = [(edge, edge_text(self.graph, *edge)) for edge in ranked]  # Serializa las candidatas.
            selected_lines, selected_edges = self._pack_lines(lines, budget)  # Aplica el límite tokenizado.
            covered = set()  # Palabras de consulta cubiertas por el candidato de este salto.
            for line in selected_lines:
                covered.update(query_terms(line) & terms)  # Une los términos reconocidos en cada línea seleccionada.
            coverage = len(covered) / (len(terms) or 1)  # Evita división por cero si la pregunta no tiene términos.
            rank = (coverage, len(selected_edges))  # Desempata cobertura igual con mayor número de aristas.
            if rank > best_rank:
                best_rank = rank  # Registra el salto que produjo mejor cobertura.
                selected_nodes = sorted({node for edge in selected_edges for node in edge})  # Nodos del candidato actual.
                best = GraphContext(
                    contexts=selected_lines,  # Líneas a entregar al LLM.
                    edges=selected_edges,  # Aristas del nivel ganador.
                    nodes=selected_nodes,  # Nodos del nivel ganador.
                    metadata={
                        "strategy": "adaptive",  # Método de expansión adaptativa.
                        "hops": hop,  # Nivel que ganó según cobertura/numero de aristas.
                        "max_hops": max_hops,  # Profundidad máxima examinada.
                        "decay": decay,  # Factor de atenuación por salto.
                        "seeds": len(seeds),  # Semillas iniciales.
                        "edge_anchors": len(anchors),  # Coincidencias directas de aristas.
                        "candidates": len(candidates),  # Aristas del nivel ganador antes del presupuesto.
                        "query_coverage": coverage,  # Proporción de términos cubiertos.
                        "graph_nodes": len(self.nodes),  # Tamaño del grafo original.
                        "graph_edges": len(self.edges),  # Número de relaciones de origen.
                    },
                )
        return best  # Mejor contexto encontrado hasta max_hops.
