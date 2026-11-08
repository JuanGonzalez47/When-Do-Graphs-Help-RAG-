"""Construcción, transformación y medición estructural de grafos."""

from __future__ import annotations

import re  # Limpia nombres, busca patrones de metadatos y separa evidencia.
import unicodedata  # Normaliza distintas representaciones Unicode de entidades.
from collections import deque  # Cola eficiente para búsquedas en anchura.
from pathlib import Path  # Acepta rutas de archivo como texto o Path.
from typing import Any, Iterable, Sequence  # Tipos de triples y colecciones flexibles.

import networkx as nx  # Implementa grafos y algoritmos de estructura/conectividad.

ARTICLES = {"the", "a", "an"}  # Artículos iniciales que se ignoran al normalizar entidades.


def normalize_entity(name: str) -> str:
    """Convierte nombres de entidad a una forma canónica comparable."""
    text = unicodedata.normalize("NFKC", str(name)).strip().lower()  # Unifica Unicode, espacios y mayúsculas.
    text = re.sub(r"[^\w\s]", " ", text)  # Sustituye puntuación por separadores.
    text = re.sub(r"\s+", " ", text).strip()  # Reduce varios espacios a uno.
    words = text.split()  # Separa tokens para limpiar artículos iniciales.
    while words and words[0] in ARTICLES:
        words.pop(0)  # "The character" y "character" se convierten en la misma clave.
    return " ".join(words)  # Conserva el orden de las palabras del nombre.


def normalize_relation(name: str) -> str:
    """Aplica a los predicados la misma normalización que a las entidades."""
    return normalize_entity(name)  # Mantiene comparables las etiquetas de relación.


METADATA_MARKERS = (  # Frases comunes del encabezado/créditos de textos Gutenberg.
    "gutenberg",
    "distributed proofreading",
    "project gutenberg",
    "online distributed proofreading",
    "produced by",
    "pgdp",
    "ebook",
)


def is_metadata_entity(name: str, source: str = "") -> bool:
    """Detecta nombres que probablemente provienen de metadatos y no del relato."""
    key = normalize_entity(name)  # Compara el nombre en forma canónica.
    if not key:
        return False  # Un nombre vacío por sí solo no activa el filtro.
    if key.isdigit():
        return True  # Números aislados suelen ser IDs extraídos del encabezado.
    if re.fullmatch(r"chunk( id)? \d+", key):
        return True  # Rechaza referencias generadas al ID del fragmento.
    if key == "corpus" or re.fullmatch(r"corpus (novel )?\d+", key):
        return True  # Rechaza etiquetas automáticas del nombre del corpus.
    if any(marker in key for marker in METADATA_MARKERS):
        return True  # Elimina entidades que contienen marcadores de créditos/archivo.
    source_digits = re.sub(r"\D", "", str(source))  # Extrae dígitos del ID de fuente para comparar.
    if source_digits and source_digits in key.split():
        return True  # Un número del nombre de archivo no debe convertirse en entidad.
    return False  # No se encontró una regla conocida de metadatos.


def is_metadata_triple(head: str, relation: str, tail: str, source: str = "") -> bool:
    """Marca triples cuyo extremo o predicado parece generado desde metadatos."""
    if is_metadata_entity(head, source) or is_metadata_entity(tail, source):
        return True  # Basta un extremo artificial para descartar la arista.
    relation_key = normalize_relation(relation)  # Normaliza la etiqueta antes de compararla.
    return relation_key in {"contains chunk", "corresponds to"}  # Relaciones de estructura del pipeline, no del relato.


def as_triple(item: Any) -> tuple[str, str, str]:
    """Interpreta un triple como diccionario o secuencia de tres elementos."""
    if isinstance(item, dict):
        return (
            str(item.get("head", item.get("source", ""))),  # Admite alias de extremo inicial.
            str(item.get("relation", item.get("predicate", ""))),  # Admite alias del predicado.
            str(item.get("tail", item.get("target", ""))),  # Admite alias de extremo final.
        )
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 3:
        return str(item[0]), str(item[1]), str(item[2])  # Usa los tres primeros elementos de la secuencia.
    raise TypeError(f"Cannot interpret triple: {item!r}")  # Informa el valor incompatible al extractor.


def parse_evidence_triple(text: str) -> tuple[str, str, str] | None:
    """Divide evidencia escrita como tupla de texto en sujeto, relación y objeto."""
    body = str(text).strip().rstrip(".")  # Quita espacios y punto final opcional.
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1]  # Retira paréntesis exteriores si están presentes.
    parts = [part.strip() for part in body.split(",")]  # Separa posibles componentes por coma.
    if len(parts) < 3:
        return None  # Menos de tres elementos no forman un triple.
    head = parts[0]  # Primer componente: sujeto/extremo inicial.
    tail = parts[-1]  # Último componente: objeto/extremo final.
    relation = ",".join(parts[1:-1]).strip()  # Une comas internas que pertenecen a la relación.
    if not head or not relation or not tail:
        return None  # Rechaza triples incompletos.
    return head, relation, tail  # Devuelve texto original ya separado.


def graph_from_triples(
    triples: Iterable[Any],
    directed: bool = False,
) -> nx.Graph:
    """Construye un grafo y combina etiquetas cuando se repite una pareja de nodos."""
    graph: nx.Graph = nx.MultiDiGraph() if directed else nx.Graph()  # El protocolo actual usa grafo simple no dirigido.
    for item in triples:
        head, relation, tail = as_triple(item)  # Acepta las distintas formas de almacenar un triple.
        head_key = normalize_entity(head)  # Canoniza nodo inicial para unir alias.
        tail_key = normalize_entity(tail)  # Canoniza nodo final con la misma regla.
        if not head_key or not tail_key or head_key == tail_key:
            continue  # Ignora extremos vacíos o una arista que se reduciría a auto-bucle.
        for node in (head_key, tail_key):
            if node not in graph:
                graph.add_node(node, label=node)  # Conserva una etiqueta legible junto al ID canónico.
        relation_key = normalize_relation(relation)  # Normaliza el predicado que se guardará como arista.
        if graph.has_edge(head_key, tail_key):
            relations = graph[head_key][tail_key].setdefault("relations", [])  # Lista compartida de predicados.
            if relation_key and relation_key not in relations:
                relations.append(relation_key)  # Evita repetir la misma etiqueta.
            graph[head_key][tail_key]["weight"] = graph[head_key][tail_key].get("weight", 1) + 1  # Cuenta triples sobre la pareja.
        else:
            graph.add_edge(
                head_key,
                tail_key,
                relations=[relation_key] if relation_key else [],  # Mantiene el predicado, si no quedó vacío.
                weight=1,  # Primera observación de esta pareja.
            )
    return graph  # Devuelve el grafo con nodos y relaciones agregadas.


def to_simple_graph(graph: nx.Graph) -> nx.Graph:
    """Devuelve copia simple no dirigida para los algoritmos del proyecto."""
    if graph.is_directed():
        return nx.Graph(graph)  # Convierte orientación a conectividad no dirigida.
    return graph.copy()  # Evita modificar el objeto original.


def remove_edges(graph: nx.Graph) -> nx.Graph:
    """Conserva nodos y atributos, eliminando todas las aristas."""
    stripped = nx.Graph()  # Crea un grafo vacío del tipo simple/no dirigido.
    for node, attrs in graph.nodes(data=True):
        stripped.add_node(node, **attrs)  # Copia nodos/atributos para una ablación nodes-only.
    return stripped  # El resultado no incluye relaciones.


def rewire_edges(
    graph: nx.Graph,
    seed: int = 42,
    swap_fraction: float = 0.1,
    max_tries_factor: int = 100,
) -> nx.Graph:
    """Reasigna extremos procurando conservar grados; no preserva etiquetas de arista."""
    rewired = to_simple_graph(graph)  # Copia el grafo como estructura simple.
    edge_count = rewired.number_of_edges()  # Número usado para definir intentos de intercambio.
    if edge_count < 4 or swap_fraction <= 0:
        return rewired  # Con tan pocas aristas no es viable un doble intercambio.
    n_swap = max(1, int(edge_count * swap_fraction))  # Cantidad entera mínima a intentar.
    try:
        nx.double_edge_swap(
            rewired,  # La operación cambia la topología de la copia.
            nswap=n_swap,  # Número solicitado de intercambios válidos.
            max_tries=n_swap * max_tries_factor,  # Límite para evitar intentos infinitos.
            seed=seed,  # Fija aleatoriedad y permite reproducir el resultado.
        )
    except (nx.NetworkXAlgorithmError, nx.NetworkXError):
        pass  # NetworkX puede fallar en grafos que no admiten los swaps solicitados.
    return rewired  # Devuelve el estado alcanzado; los llamadores deben verificar cambios.


def random_edges(
    graph: nx.Graph,
    seed: int = 42,
    swap_fraction: float = 0.1,
) -> nx.Graph:
    """Permuta aristas con doble intercambio manteniendo grados cuando es posible."""
    randomized = to_simple_graph(graph)  # No altera el grafo entregado.
    edge_count = randomized.number_of_edges()  # Cantidad base para calcular intercambios.
    if edge_count < 4:
        return randomized  # Un intercambio requiere suficientes aristas distintas.
    n_swap = max(1, int(edge_count * swap_fraction))  # Convierte fracción a cantidad de swaps.
    try:
        nx.double_edge_swap(
            randomized,  # Aplica cambios sobre la copia del grafo.
            nswap=n_swap,  # Número objetivo de cambios exitosos.
            max_tries=n_swap * 100,  # Limita intentos de búsqueda de cambios válidos.
            seed=seed,  # Hace determinista la aleatoriedad.
        )
    except (nx.NetworkXAlgorithmError, nx.NetworkXError):
        pass  # Devuelve el grafo alcanzado si el algoritmo no logra todos los intercambios.
    return randomized  # No copia los atributos de arista, solo la topología.


def subgraph_by_nodes(graph: nx.Graph, nodes: Iterable[str]) -> nx.Graph:
    """Extrae una copia inducida por los nodos solicitados que sí existen."""
    keep = [node for node in nodes if node in graph]  # Descarta IDs que no pertenecen al grafo.
    return graph.subgraph(keep).copy()  # Conserva aristas entre los nodos retenidos.


def subgraph_by_edges(graph: nx.Graph, edges: Iterable[tuple[str, str]]) -> nx.Graph:
    """Construye subgrafo copiando las aristas elegidas y sus atributos."""
    selected = nx.Graph()  # El subgrafo experimental usa estructura no dirigida.
    for head, tail in edges:
        if graph.has_edge(head, tail):
            attrs = dict(graph[head][tail])  # Copia peso y predicados para no compartir diccionarios.
            selected.add_edge(head, tail, **attrs)  # Inserta la arista seleccionada.
            for node in (head, tail):
                selected.nodes[node].update(dict(graph.nodes[node]))  # Copia etiquetas y propiedades de los extremos.
    return selected  # Los nodos aislados no se añaden si no aparecen en una arista.


def prune_edges_random(graph: nx.Graph, keep_fraction: float = 0.5, seed: int = 42) -> nx.Graph:
    """Conserva una fracción aleatoria reproducible de aristas."""
    if not 0 <= keep_fraction <= 1:
        raise ValueError("keep_fraction must be in [0, 1]")  # Rechaza fracciones inválidas.
    import random  # Import local para que el módulo no dependa de este helper al cargarse.

    rng = random.Random(seed)  # Generador aislado para que no cambie el estado aleatorio global.
    edges = list(graph.edges())  # Materializa aristas para mezclarlas.
    rng.shuffle(edges)  # Aleatoriza el orden de selección con la semilla dada.
    keep = int(round(len(edges) * keep_fraction))  # Número de aristas a conservar.
    return subgraph_by_edges(graph, edges[:keep])  # Copia las primeras aristas elegidas.


def prune_edges_centrality(graph: nx.Graph, keep_fraction: float = 0.5) -> nx.Graph:
    """Conserva aristas de mayor centralidad de intermediación."""
    if not 0 <= keep_fraction <= 1:
        raise ValueError("keep_fraction must be in [0, 1]")  # Solo acepta fracciones de cero a uno.
    simple = to_simple_graph(graph)  # Centralidad se calcula sobre grafo no dirigido simple.
    if simple.number_of_edges() == 0:
        return remove_edges(simple)  # Evita pedir centralidad de un grafo sin aristas.
    centrality = nx.edge_betweenness_centrality(simple)  # Mide cuánto pasan rutas mínimas por cada arista.
    ranked = sorted(centrality, key=lambda edge: centrality[edge], reverse=True)  # Ordena de más a menos central.
    keep = max(1, int(round(len(ranked) * keep_fraction)))  # Mantiene al menos una arista si hay alguna.
    return subgraph_by_edges(simple, ranked[:keep])  # Devuelve la copia con aristas priorizadas.


def graph_metrics(graph: nx.Graph) -> dict[str, float]:
    """Calcula conteos, densidad, agrupamiento y componentes del grafo."""
    simple = to_simple_graph(graph)  # Estandariza la estructura para métricas comparables.
    nodes = simple.number_of_nodes()  # Total de nodos del grafo.
    edges = simple.number_of_edges()  # Total de parejas conectadas.
    degrees = [degree for _, degree in simple.degree()]  # Grado de cada nodo.
    avg_degree = sum(degrees) / nodes if nodes else 0.0  # Evita dividir por cero en grafo vacío.
    avg_clustering = nx.average_clustering(simple) if nodes > 1 else 0.0  # Coeficiente de cierre de triángulos.
    components = nx.number_connected_components(simple) if nodes else 0  # Cantidad de componentes conexos.
    largest = max((len(component) for component in nx.connected_components(simple)), default=0)  # Tamaño de componente mayor.
    return {
        "node_count": nodes,  # Tamaño de vértices.
        "edge_count": edges,  # Tamaño de relaciones.
        "average_degree": float(avg_degree),  # Conexiones medias por nodo.
        "average_clustering": float(avg_clustering),  # Tendencia a formar triángulos.
        "density": float(nx.density(simple)) if nodes > 1 else 0.0,  # Fracción de aristas posibles que existen.
        "components": components,  # Fragmentos independientes del grafo.
        "largest_component": largest,  # Nodos del componente más grande.
        "isolated_nodes": sum(1 for degree in degrees if degree == 0),  # Nodos sin relaciones.
    }


def evidence_coverage(
    graph: nx.Graph,
    triples: Iterable[Any],
    require_relation: bool = False,
) -> dict[str, float]:
    """Mide si evidencia de referencia aparece conectada y con su predicado."""
    simple = to_simple_graph(graph)  # Usa la misma representación que la extracción.
    total = 0  # Cantidad de triples de referencia evaluados.
    exact = 0  # Cantidad que además conserva el predicado exigido.
    structural = 0  # Cantidad cuyos dos extremos aparecen unidos.
    for item in triples:
        head, relation, tail = as_triple(item)  # Normaliza las formas de evidencia admitidas.
        head_key = normalize_entity(head)  # Clave comparable al nodo del grafo.
        tail_key = normalize_entity(tail)  # Clave comparable al otro extremo.
        relation_key = normalize_relation(relation)  # Clave comparable a las etiquetas almacenadas.
        total += 1  # Cuenta incluso triples ausentes para que bajen la cobertura.
        if head_key not in simple or tail_key not in simple:
            continue  # No hay cobertura si falta cualquiera de los extremos.
        if not simple.has_edge(head_key, tail_key):
            continue  # Los dos nodos deben estar unidos por una arista.
        structural += 1  # Coincidencia topológica, independientemente de etiqueta.
        relations = simple[head_key][tail_key].get("relations", [])  # Predicados asociados a la arista.
        if not require_relation or relation_key in relations:
            exact += 1  # Sin requisito de predicado, o con predicado exacto.
    denominator = total or 1  # Evita división por cero para una lista vacía.
    return {
        "triples": total,  # Denominador real de triples consultados.
        "structural_coverage": structural / denominator,  # Proporción de pares conectados.
        "relation_coverage": exact / denominator,  # Proporción con etiqueta cuando se solicita.
    }


def reachability(graph: nx.Graph, sources: Iterable[str], targets: Iterable[str]) -> float:
    """Calcula qué proporción de destinos conocidos es alcanzable desde las fuentes."""
    simple = to_simple_graph(graph)  # Traversal de conectividad no dirigida.
    source_keys = [normalize_entity(node) for node in sources if normalize_entity(node) in simple]  # Fuentes presentes.
    target_keys = [normalize_entity(node) for node in targets if normalize_entity(node) in simple]  # Destinos presentes.
    if not target_keys:
        return 0.0  # Sin destinos conocidos no hay cobertura medible.
    reached: set[str] = set()  # Nodos descubiertos desde todas las fuentes.
    for source in source_keys:
        visited = {source}  # Cada búsqueda evita volver a visitar nodos.
        queue = deque([source])  # Búsqueda en anchura desde una fuente.
        while queue:
            node = queue.popleft()  # Extrae el siguiente nodo por explorar.
            for neighbor in simple.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)  # Marca antes de insertar para evitar duplicados en la cola.
                    queue.append(neighbor)  # Explora en una iteración posterior.
        reached.update(visited)  # Une los componentes alcanzables desde cada fuente.
    return sum(1 for target in target_keys if target in reached) / len(target_keys)  # Fracción de destinos conectados.


def graph_to_records(graph: nx.Graph) -> dict[str, Any]:
    """Convierte NetworkX a JSON simple de nodos y aristas."""
    return {
        "nodes": [
            {"id": node, **attrs} for node, attrs in graph.nodes(data=True)  # Incluye atributos de nodo junto a su ID.
        ],
        "edges": [
            {"source": head, "target": tail, **attrs}
            for head, tail, attrs in graph.edges(data=True)  # Guarda extremos y metadatos de arista.
        ],
    }


def graph_from_records(records: dict[str, Any]) -> nx.Graph:
    """Reconstruye un grafo desde el formato JSON serializado por graph_to_records."""
    graph = nx.Graph()  # Carga como grafo simple no dirigido.
    for node in records.get("nodes", []):
        attrs = {key: value for key, value in node.items() if key != "id"}  # Separa ID de atributos.
        graph.add_node(str(node["id"]), **attrs)  # Convierte IDs a cadenas para normalizar tipos.
    for edge in records.get("edges", []):
        attrs = {
            key: value
            for key, value in edge.items()
            if key not in {"source", "target"}  # Los extremos son argumentos, no atributos.
        }
        relations = attrs.get("relations")  # Recupera etiquetas si existen.
        if relations is not None and not isinstance(relations, list):
            attrs["relations"] = [str(relations)]  # Normaliza una etiqueta suelta al esquema de lista.
        graph.add_edge(str(edge["source"]), str(edge["target"]), **attrs)  # Inserta la conexión y sus atributos.
    return graph  # Devuelve la estructura reconstruida.


def save_graph(path: str | Path, graph: nx.Graph) -> None:
    """Serializa un grafo a JSON usando la representación del proyecto."""
    from .io import write_json  # Import local evita dependencia circular al cargar el módulo.

    write_json(path, graph_to_records(graph))  # Escritura JSON con creación de carpetas.


def load_graph(path: str | Path) -> nx.Graph:
    """Lee un JSON de grafo y lo reconstruye como NetworkX."""
    from .io import read_json  # Import local mantiene el mismo desacoplamiento que save_graph.

    return graph_from_records(read_json(path))  # Decodifica primero y después reconstruye nodos/aristas.
