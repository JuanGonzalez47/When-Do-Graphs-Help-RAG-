#!/usr/bin/env python3
"""Genera condiciones controladas de Q3 y experimentos de profundidad de Q4.

Q3 cambia contenido/estructura del grafo y mantiene la profundidad y longitud
del contexto comparables entre intervenciones. Q4 compara profundidades de
recuperación en una muestra de desarrollo congelada y estratificada. Los
checkpoints y huellas permiten reanudar sin mezclar configuraciones distintas.
"""

# Evita evaluar anotaciones de tipos al importar este archivo.
from __future__ import annotations

# argparse define opciones de terminal; hashlib calcula huellas de archivos/datos.
import argparse
import hashlib
# json serializa manifiestos y predicciones; random genera selecciones reproducibles.
import json
import random
# sys modifica la búsqueda de módulos; time mide el tiempo transcurrido.
import sys
import time
# defaultdict crea grupos vacíos automáticamente al agregar una nueva clave.
from collections import defaultdict
# Path construye rutas de archivos de forma segura.
from pathlib import Path
# Any y Callable describen tipos flexibles y funciones recibidas como argumento.
from typing import Any, Callable

# NetworkX representa los grafos y sus nodos/aristas.
import networkx as nx
# tiktoken calcula la longitud real de los contextos con el tokenizador del protocolo.
import tiktoken

# La raíz se deriva de la carpeta de este script, no del directorio desde donde se llama.
ROOT = Path(__file__).resolve().parents[1]
# Agrega research/ al path para importar el código compartido src.
sys.path.insert(0, str(ROOT / "research"))

# Helpers comunes: serialización de contextos y truncado según tokens.
from src.budget import format_contexts, truncate_to_tokens  
# Carga recursos del corpus y prepara el baseline RAG.
from src.conditions import ExperimentResources, _rag_contexts, load_resources 
# LEVELS contiene los cuatro tipos de pregunta; load_sample lee los 400 IDs congelados.
from src.data import LEVELS, load_sample 
# Reutiliza el formato de predicción y la llamada de generación del modelo.
from src.generation import build_prediction, generate_answer 
# GraphIndex implementa los métodos de recuperación de contexto desde el grafo.
from src.graph_retrieval import GraphIndex  
# Helpers de lectura/escritura JSONL y JSON usados por checkpoints y configuración.
from src.io import append_jsonl, read_jsonl, write_json  
# Valores por omisión de Ollama y cliente compatible con su API.
from src.llm import DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL, LLMClient  
# Carpeta que contiene los grafos construidos previamente.
from src.paths import GRAPHS  
# Prompts de generación; también se calcula su hash para identificar el protocolo.
from src.prompts import GENERATION_SYSTEM, GENERATION_USER  

# Nombre exacto del tokenizador utilizado para contar el contexto.
TOKENIZER_NAME = "cl100k_base"
# Q3 y Q4 usan un máximo solicitado de contexto de 4.096 tokens.
Q3_BUDGET = 4096
Q4_BUDGET = 4096
# Variantes que responden la pregunta Q3 sobre el aporte de relaciones de grafo.
Q3_CONDITIONS = (
    "graph_intact",
    "graph_nodes_only",
    "graph_hidden_predicates",
    "graph_permuted_predicates",
    "graph_rewired_topology",
    "rag_reference",
)
# Métodos de recuperación probados por Q4.
Q4_METHODS = ("graph_full", "graph_greedy", "graph_adaptive")
# Profundidades/hops evaluados por Q4: los extremos 1 y 6 están incluidos.
Q4_DEPTHS = tuple(range(1, 7))
# Semilla fija para derivaciones deterministas de IDs y particiones.
Q4_SPLIT_SEED = 2026
# Archivo donde se congela la partición de desarrollo y confirmación de Q4.
Q4_SPLIT_PATH = ROOT / "research" / "configs" / "q4_split_2026.json"
# Predicciones finales y checkpoints se guardan bajo esta carpeta.
DEFAULT_OUTPUT_ROOT = ROOT / "research" / "results" / "evaluation_raw" / "q3_q4" / "generation"
# Manifiestos de ejecución se guardan junto a los datos brutos de Q3/Q4.
DEFAULT_RUNS_ROOT = ROOT / "research" / "results" / "evaluation_raw" / "q3_q4" / "generation_runs"
# Máximo de tokens que puede producir el modelo por respuesta.
MAX_OUTPUT_TOKENS = 512
# Temperatura cero busca reducir variación aleatoria de la generación.
GENERATION_TEMPERATURE = 0.0
# Q3 usa siempre dos saltos para elegir el subgrafo inicial compartido.
Q3_HOPS = 2
# Fracción de aristas que se intenta intercambiar en cada intervención rewired.
REWIRE_SWAP_FRACTION = 0.1
# Identificador textual del protocolo; cambiarlo marca una versión experimental distinta.
PROTOCOL_VERSION = "q3q4-2026-11-05-v1"


class RewireUnavailableError(RuntimeError):
    """Error esperado cuando el subgrafo no permite intercambiar aristas."""

    # No agrega comportamiento nuevo; distingue este fallo de otros errores.
    pass


def _stable_seed(question_id: str, condition: str) -> int:
    """Deriva una semilla reproducible de una pregunta y una condición."""
    # El texto contiene los componentes que deben fijar la secuencia pseudoaleatoria.
    digest = hashlib.sha256(f"{Q4_SPLIT_SEED}:{question_id}:{condition}".encode()).digest()
    # Usa los primeros cuatro bytes del SHA-256 como entero de semilla.
    return int.from_bytes(digest[:4], "big")


def _rendered_tokens(contexts: list[str], encoding: tiktoken.Encoding) -> int:
    """Cuenta tokens después de aplicar el formato que verá el prompt."""
    # format_contexts agrega la misma numeración/separadores usados al generar.
    return len(encoding.encode(format_contexts(contexts)))


def _pad_to_tokens(
    context: str,
    target: int,
    encoding: tiktoken.Encoding,
) -> tuple[str, int]:
    """Añade texto de relleno hasta alcanzar exactamente el número objetivo de tokens."""
    # El formato final cuenta envolturas del contexto además del contenido mismo.
    contexts = [context]
    # Mide el texto original antes de decidir si hay que añadir relleno.
    current = _rendered_tokens(contexts, encoding)
    # Un texto que ya excede el objetivo no se puede reparar añadiendo tokens.
    if current > target:
        raise ValueError(f"Context has {current} tokens, exceeding target {target}")
    # Si ya tiene longitud correcta, devuelve el contenido sin tocarlo.
    if current == target:
        return context, 0

    # Diferencia entre longitud deseada y longitud actual.
    needed = target - current
    # Se prueban marcadores sencillos; siguen siendo texto visible para el LLM.
    for marker in (" [PAD]", " [MASK]"):
        # Cada marcador puede tokenizarse en uno o más tokens.
        marker_tokens = len(encoding.encode(marker))
        # Evita dividir por cero si el tokenizador descarta el marcador.
        if marker_tokens:
            # Cantidad de marcadores completos y tokens faltantes aproximados.
            repetitions, remainder = divmod(needed, marker_tokens)
            # Prueba sufijos que ayudan a ajustar la tokenización exacta.
            for single_token in (" x", " y", " z", " q"):
                # Forma un candidato con repeticiones y un ajuste final.
                candidate = (
                    context
                    + marker * repetitions
                    + single_token * remainder
                )
                # Devuelve solo cuando el conteo real coincide exactamente.
                if _rendered_tokens([candidate], encoding) == target:
                    return candidate, needed
    # Último intento: agrega un sufijo repetido para cubrir los tokens faltantes.
    candidate = context + " x" * needed
    # Comprueba el conteo de nuevo porque tokenización puede depender del texto vecino.
    if _rendered_tokens([candidate], encoding) == target:
        return candidate, needed

    # No se debe aceptar en silencio una condición Q3 con longitud despareja.
    raise ValueError(
        f"Could not pad serialized context from {current} to exactly {target} tokens"
    )


def _edge_lines(
    graph: nx.Graph,
    edges: list[tuple[str, str]],
    relation_mode: str,
    seed: int,
) -> tuple[list[str], dict[str, Any]]:
    """Convierte aristas en texto y opcionalmente oculta o permuta sus relaciones."""
    # Lee la lista de predicados guardada como atributo de cada arista.
    bundles = [
        list(graph[head][tail].get("relations", []))
        for head, tail in edges
    ]
    # Cuenta cuántas aristas terminan con predicados distintos tras la permutación.
    changed_edges = 0
    # Un generador local evita afectar aleatoriedad global del proceso.
    rng = random.Random(seed)
    # Solo se permutan predicados cuando la condición lo solicita y hay varias aristas.
    if relation_mode == "permuted" and len(bundles) > 1:
        # Copia la asociación original para comparar después qué cambió.
        original = [tuple(bundle) for bundle in bundles]
        # Mezcla hasta diez veces para intentar evitar la permutación identidad.
        for _ in range(10):
            rng.shuffle(bundles)
            # Una permutación diferente rompe asociación predicado-extremos.
            if [tuple(bundle) for bundle in bundles] != original:
                break
        # Cuenta los paquetes cuya posición final difiere de la original.
        changed_edges = sum(
            tuple(before) != tuple(after)
            for before, after in zip(original, bundles)
        )

    # Construye una línea de texto por arista del grafo seleccionado.
    lines = []
    for (head, tail), relations in zip(edges, bundles):
        # En la ablación hidden se mantiene el lugar de la relación, pero se enmascara.
        if relation_mode == "hidden":
            relation_text = "[MASK]"
        else:
            # Si hay varias relaciones en una arista, se separan con punto y coma.
            relation_text = " ; ".join(relations)
        # Serializa los extremos y predicado en la forma que leerá el modelo.
        lines.append(f"{head} -- {relation_text} -- {tail}")
    # Devuelve texto y metadatos de la intervención.
    return lines, {
        "relation_mode": relation_mode,
        "changed_relation_bundles": changed_edges,
    }


def _rewire_selected_edges(
    edges: list[tuple[str, str]],
    seed: int,
) -> tuple[list[tuple[str, str]], int]:
    """Intercambia aristas no dirigidas y conserva el grado de cada nodo."""
    # Ordenar los extremos representa cada arista sin importar su orientación.
    edge_set = {tuple(sorted(edge)) for edge in edges}
    # No se acepta entrada con duplicados porque alteraría el número de aristas.
    if len(edge_set) != len(edges):
        raise ValueError("Selected subgraph unexpectedly contains duplicate edges")
    # Un doble intercambio necesita como mínimo dos aristas y cuatro extremos distintos.
    if len(edge_set) < 4:
        raise RewireUnavailableError(
            f"Cannot rewire selected subgraph with only {len(edge_set)} edges"
        )

    # La semilla mantiene el mismo resultado para la pregunta si se repite el experimento.
    rng = random.Random(seed)
    # Se busca intercambiar aproximadamente el diez por ciento, como mínimo una vez.
    target_swaps = max(1, int(len(edge_set) * REWIRE_SWAP_FRACTION))
    # Cuenta intercambios completados, intentos y el límite para evitar bucles eternos.
    swaps = 0
    attempts = 0
    max_attempts = max(100, target_swaps * 100)
    # Se detiene al alcanzar el objetivo o agotar intentos.
    while swaps < target_swaps and attempts < max_attempts:
        # Cada intento examina dos aristas elegidas al azar.
        attempts += 1
        first, second = rng.sample(sorted(edge_set), 2)
        # Separa los cuatro extremos de las aristas seleccionadas.
        a, b = first
        c, d = second
        # Cuatro nodos diferentes son necesarios para no cambiar grados ni crear lazos.
        if len({a, b, c, d}) != 4:
            continue
        # Alterna cuál extremo de la segunda arista se empareja con el primero.
        if rng.randrange(2):
            c, d = d, c
        # Las nuevas parejas forman el doble intercambio propuesto.
        new_first = tuple(sorted((a, c)))
        new_second = tuple(sorted((b, d)))
        # Rechaza lazos, duplicados entre las nuevas aristas o enlaces que ya existían.
        if (
            new_first == new_second
            or new_first in edge_set
            or new_second in edge_set
        ):
            continue
        # Quita las aristas antiguas y agrega las dos nuevas.
        edge_set.remove(first)
        edge_set.remove(second)
        edge_set.add(new_first)
        edge_set.add(new_second)
        # El doble intercambio válido preserva el número de aristas.
        swaps += 1

    # No se puede afirmar que cambió la topología si ningún intercambio tuvo éxito.
    if swaps == 0:
        raise RewireUnavailableError(
            f"No valid degree-preserving edge swap found in {attempts} attempts"
        )
    # Ordena para que el resultado serializado sea estable y devuelve el contador real.
    return sorted(edge_set), swaps


def _q3_graph_variants(
    index: GraphIndex,
    question: str,
    budget: int,
    question_id: str,
    encoding: tiktoken.Encoding,
) -> dict[str, tuple[list[str], dict[str, Any]]]:
    """Construye ablaciones Q3 del mismo subgrafo y empareja su total de tokens."""
    # El subgrafo base se recupera una sola vez con el método graph_full a dos saltos.
    result = index.full_context(question, budget=budget, hops=Q3_HOPS)
    # Se accede a las relaciones originales para construir cada transformación.
    graph = index.graph
    # Copia las aristas seleccionadas por el recuperador para no cambiar el grafo original.
    edges = list(result.edges)
    # Sin aristas no se pueden formar las intervenciones que dependen de relaciones.
    if not edges:
        raise ValueError(f"{question_id}: graph retrieval selected no edges")

    # Si las representaciones no caben juntas, se reduce el mismo subgrafo base.
    while edges:
        # La misma semilla de pregunta gobierna permutación de predicados.
        seed = _stable_seed(question_id, "predicates")
        # Versión intacta: extremos y relaciones del grafo original.
        intact_lines, _ = _edge_lines(graph, edges, "intact", seed)
        # Versión oculta: mismos extremos con la etiqueta de relación tapada.
        hidden_lines, _ = _edge_lines(graph, edges, "hidden", seed)
        # Versión permutada: cambia qué paquete de relaciones corresponde a cada arista.
        permuted_lines, permuted_meta = _edge_lines(graph, edges, "permuted", seed)
        # Cambia la conectividad manteniendo la secuencia de grados.
        rewired_edges, swap_count = _rewire_selected_edges(
            edges, _stable_seed(question_id, "rewire")
        )
        # Copia los paquetes de predicados para reasignarlos a la nueva topología.
        relation_bundles = [
            list(graph[head][tail].get("relations", []))
            for head, tail in edges
        ]
        # Importante: esta condición altera topología y asignación predicado-arista.
        rng = random.Random(_stable_seed(question_id, "rewired_predicates"))
        # Reasigna predicados independientemente de los nuevos extremos.
        rng.shuffle(relation_bundles)
        # Construye las líneas con nuevas parejas y predicados mezclados.
        rewired_lines = [
            f"{head} -- {' ; '.join(relations)} -- {tail}"
            for (head, tail), relations in zip(rewired_edges, relation_bundles)
        ]
        # Nodos presentes en las aristas originales seleccionadas.
        selected_nodes = sorted({node for edge in edges for node in edge})
        # Todas las ablaciones se derivan de la misma selección base de la pregunta.
        candidates = {
            "graph_intact": ("\n".join(intact_lines), edges),
            "graph_hidden_predicates": ("\n".join(hidden_lines), edges),
            "graph_permuted_predicates": ("\n".join(permuted_lines), edges),
            "graph_rewired_topology": ("\n".join(rewired_lines), rewired_edges),
            "graph_nodes_only": ("\n".join(selected_nodes), []),
        }
        # Mide el conteo real de cada representación antes de añadir relleno.
        counts = {
            name: _rendered_tokens([body], encoding)
            for name, (body, _) in candidates.items()
        }
        # Se elige el contexto natural más largo para igualar todas las condiciones.
        target = max(counts.values())
        # Si cabe bajo el límite, las variantes comparten tamaño exacto.
        if target <= budget:
            break
        # Si no cabe, quita una arista y vuelve a formar todas las variantes.
        edges.pop()

    # Si se agotaron las aristas, ni siquiera la versión reducida cabe en el presupuesto.
    if not edges:
        raise ValueError(
            f"{question_id}: transformed Q3 graph contexts cannot fit {budget} tokens"
        )

    # Diccionario final: nombre de condición -> textos de contexto y metadatos.
    variants: dict[str, tuple[list[str], dict[str, Any]]] = {}
    # Recorre las cinco variantes de grafo, sin incluir aquí el baseline RAG.
    for name, (body, condition_edges) in candidates.items():
        # Añade relleno para que el contexto completo tenga longitud exactamente igual.
        padded, padding_tokens = _pad_to_tokens(body, target, encoding)
        # Una discrepancia es un error de protocolo; no debe ocultarse.
        if _rendered_tokens([padded], encoding) != target:
            raise AssertionError(f"{question_id}/{name}: token matching failed")
        # Guarda contenido y datos para auditoría junto con cada condición.
        variants[name] = (
            [padded],
            {
                # Identifica el experimento y nombre de la intervención.
                "experiment": "Q3",
                "condition": name,  # Nombre de la variante generada.
                "requested_budget": budget,  # Límite máximo solicitado en tokens.
                "context_tokens": target,  # Tokens medidos del texto final.
                "matched_context_tokens": target,  # Tamaño común a todas las ablaciones de grafo.
                "padding_tokens": padding_tokens,  # Tokens añadidos para alcanzar ese tamaño.
                # Todos usan los mismos dos saltos para seleccionar el subgrafo base.
                "selected_hops": Q3_HOPS,
                # Cuenta las aristas/nodos de la variante luego de la transformación.
                "selected_edges": len(condition_edges),  # Aristas que quedan en esta condición.
                "selected_nodes": len(
                    {node for edge in condition_edges for node in edge}
                )
                if condition_edges
                else len(selected_nodes),
                # También registra el tamaño del subgrafo original usado como base.
                "base_subgraph_edges": len(edges),  # Aristas que formaban la selección inicial.
                "base_subgraph_nodes": len(selected_nodes),  # Nodos de la selección inicial.
                # Solo la variante rewired tiene un número de intercambios.
                "rewire_swaps": swap_count
                if name == "graph_rewired_topology"
                else None,
                # Solo la variante permutada tiene este contador.
                "predicate_bundles_changed": permuted_meta["changed_relation_bundles"]
                if name == "graph_permuted_predicates"
                else None,
                # Describe la serialización y permite rastrear la aleatoriedad.
                "serialization": "single_context_block_with_neutral_padding",  # Forma del texto enviado.
                "seed": _stable_seed(question_id, name),  # Permite reconstruir transformaciones aleatorias.
            },
        )
    # El llamador selecciona la condición y registra su metadato en la predicción.
    return variants


def _q3_rag_context(
    record: dict[str, Any],
    resources: ExperimentResources,
    encoding: tiktoken.Encoding,
) -> tuple[list[str], dict[str, Any]]:
    """Recupera contexto RAG de texto para usarlo como referencia externa en Q3."""
    # Busca chunks del corpus que correspondan a la pregunta y a su fuente.
    contexts, metadata = _rag_contexts(
        str(record["question"]),
        str(record["source"]),
        Q3_BUDGET,
        resources,
    )
    # El baseline se guarda como un bloque para compararlo con contextos de grafo.
    body = "\n\n".join(contexts)
    # Cuenta los tokens del bloque ya serializado, no el número de caracteres.
    actual_tokens = _rendered_tokens([body], encoding)
    # El recuperador puede devolver un bloque algo mayor por los delimitadores.
    if actual_tokens > Q3_BUDGET:
        # Primer intento: truncar con el tokenizador configurado.
        body = truncate_to_tokens(body, Q3_BUDGET, TOKENIZER_NAME)
        # Vuelve a contar después del truncado para comprobar el límite real.
        actual_tokens = _rendered_tokens([body], encoding)
        # El formato agregado también consume tokens; quita tokens hasta quedar dentro.
        while actual_tokens > Q3_BUDGET:
            # Obtiene los tokens exactos del texto actual.
            tokens = encoding.encode(body)
            # Elimina el token final y reconstruye el texto restante.
            body = encoding.decode(tokens[:-1])
            # Repite la medición porque decodificar puede modificar el conteo del borde.
            actual_tokens = _rendered_tokens([body], encoding)
    # Devuelve contexto y datos de recuperación para la auditoría de la predicción.
    return [body], {
        "experiment": "Q3",
        "condition": "rag_reference",
        "requested_budget": Q3_BUDGET,
        "context_tokens": actual_tokens,
        "retrieved_chunks": metadata["retrieved"],
        "selected_chunks": metadata["selected"],
        "serialization": "single_context_block",
    }


def _q4_context(
    index: GraphIndex,
    method: str,
    question: str,
    depth: int,
    budget: int,
    encoding: tiktoken.Encoding,
) -> tuple[list[str], dict[str, Any]]:
    """Construye contexto Q4 con método de recuperación y profundidad solicitados."""
    # Cada rama llama a una estrategia distinta sobre el mismo índice de grafo.
    if method == "graph_full":
        result = index.full_context(question, budget=budget, hops=depth)
    elif method == "graph_greedy":
        result = index.greedy_context(question, budget=budget, hops=depth)
    elif method == "graph_adaptive":
        # En adaptive, depth se traduce al límite superior de saltos permitidos.
        result = index.adaptive_context(question, budget=budget, max_hops=depth)
    else:
        # No se acepta un nombre de método que no pertenezca al protocolo Q4.
        raise ValueError(f"Unsupported Q4 method: {method}")

    # Copia los contextos/aristas devueltos para poder recortar el resultado localmente.
    contexts = list(result.contexts)
    edges = list(result.edges)
    # El formato puede sumar tokens adicionales; quita líneas/aristas desde el final.
    while contexts and _rendered_tokens(contexts, encoding) > budget:
        contexts.pop()
        if edges:
            edges.pop()
    # Verificación final del límite después de recortar.
    if _rendered_tokens(contexts, encoding) > budget:
        raise ValueError(f"Q4 context exceeds {budget} tokens after trimming")
    # Convierte los metadatos internos del recuperador a un diccionario modificable.
    metadata = result.to_metadata()
    # Añade protocolo, profundidad, límite y tamaño resultante del contexto.
    metadata.update(
        {
            "experiment": "Q4",
            "condition": method,
            "depth": depth,
            "requested_budget": budget,
            "context_tokens": _rendered_tokens(contexts, encoding),
            "selected_edges": len(edges),
            "selected_nodes": len({node for edge in edges for node in edge}),
            "serialization": "numbered_context_lines",
        }
    )
    # Retorna los contextos que recibirá el LLM y sus metadatos de medición.
    return contexts, metadata


def _make_q4_split(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Crea una partición 320/80 equilibrada por corpus y tipo de pregunta."""
    # Agrupa registros por las ocho combinaciones corpus x tipo de pregunta.
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    # Clasifica cada ID como Novel o Medical y lo añade a su estrato.
    for record in records:
        split = "novel" if str(record["id"]).startswith("Novel") else "medical"
        cells[(split, str(record["question_type"]))].append(record)
    # Se acumularán IDs separados para desarrollo y confirmación.
    development: list[str] = []
    confirmation: list[str] = []
    # Itera en orden estable sobre corpus y los cuatro niveles de pregunta.
    for split in ("novel", "medical"):
        for level in LEVELS:
            # Ordena IDs antes del muestreo para que el resultado no dependa del orden del JSON.
            cell = sorted(cells[(split, level)], key=lambda row: str(row["id"]))
            # El diseño parte de 50 preguntas en cada uno de los ocho estratos.
            if len(cell) != 50:
                raise ValueError(f"Expected 50 records in {split}/{level}, found {len(cell)}")
            # Se eligen diez IDs para confirmación con semilla derivada del estrato.
            selected = set(
                random.Random(
                    _stable_seed(f"{split}:{level}", str(Q4_SPLIT_SEED))
                ).sample([str(row["id"]) for row in cell], 10)
            )
            # Añade los diez seleccionados a la partición de confirmación.
            confirmation.extend(str(row["id"]) for row in cell if str(row["id"]) in selected)
            # Los otros cuarenta del estrato quedan para desarrollo.
            development.extend(str(row["id"]) for row in cell if str(row["id"]) not in selected)
    # Devuelve ambas listas, semilla, método de estratificación y tamaños esperados.
    return {
        "seed": Q4_SPLIT_SEED,  # Semilla que fija la selección de IDs.
        "stratification": "corpus x question_type",  # Grupos que deben quedar balanceados.
        "development_ids": development,  # Cuarenta preguntas en cada estrato.
        "confirmation_ids": confirmation,  # Diez preguntas reservadas por estrato.
        "counts": {"development": 320, "confirmation": 80},  # Totales planeados.
    }


def _load_or_create_q4_split(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Carga o crea la partición Q4 y valida identidad, cantidades y balance."""
    # Conjunto completo de IDs que debe repartirse sin omisiones ni extras.
    expected_ids = {str(record["id"]) for record in records}
    # Si ya hay partición congelada, se respeta en lugar de volver a sortear.
    if Q4_SPLIT_PATH.exists():
        split = json.loads(Q4_SPLIT_PATH.read_text(encoding="utf-8"))
    else:
        # En la primera ejecución se genera y se guarda el corte determinista.
        split = _make_q4_split(records)
        write_json(Q4_SPLIT_PATH, split)

    # Extrae las dos particiones desde el archivo.
    development = split.get("development_ids", [])
    confirmation = split.get("confirmation_ids", [])
    # Comprueba tamaño, unicidad, ausencia de solapamiento, cobertura total y semilla.
    if (
        len(development) != 320
        or len(confirmation) != 80
        or len(set(development)) != 320
        or len(set(confirmation)) != 80
        or set(development) & set(confirmation)
        or set(development) | set(confirmation) != expected_ids
        or split.get("seed") != Q4_SPLIT_SEED
    ):
        raise ValueError(f"Invalid or incompatible Q4 split file: {Q4_SPLIT_PATH}")

    # Índice para recuperar tipo/corpus de cada ID sin volver a buscar toda la lista.
    by_id = {str(record["id"]): record for record in records}
    # Confirma que el balance por estrato sea 40 para desarrollo y 10 para confirmación.
    for phase, ids in (("development", development), ("confirmation", confirmation)):
        # Contadores de cuántos registros hay por corpus y tipo.
        grouped: dict[tuple[str, str], int] = defaultdict(int)
        # Cuenta los IDs de la partición que se está validando.
        for record_id in ids:
            record = by_id[record_id]
            corpus = "novel" if record_id.startswith("Novel") else "medical"
            grouped[(corpus, str(record["question_type"]))] += 1
        # El tamaño esperado cambia según sea fase de desarrollo o confirmación.
        expected_per_cell = 40 if phase == "development" else 10
        # Si una celda difiere, la partición no respeta el diseño experimental.
        if any(grouped[(corpus, level)] != expected_per_cell for corpus in ("novel", "medical") for level in LEVELS):
            raise ValueError(f"Q4 {phase} split is not stratified as specified")
    # Devuelve la partición validada para que generación/evaluación usen los mismos IDs.
    return split


def _config_fingerprint(config: dict[str, Any]) -> str:
    """Calcula una huella estable de configuración para proteger los checkpoints."""
    # Orden y separadores fijos evitan que espacios/orden de claves cambien el hash.
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    # SHA-256 identifica la configuración exacta de esta celda experimental.
    return hashlib.sha256(canonical.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    """Calcula SHA-256 de un archivo leyendo bloques para ahorrar memoria."""
    # Crea el acumulador del digest.
    digest = hashlib.sha256()
    # Abre el archivo en modo binario para hashear sus bytes exactos.
    with path.open("rb") as handle:
        # Lee bloques de 1 MiB hasta que read() devuelva bytes vacíos.
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            # Añade cada bloque al digest incremental.
            digest.update(block)
    # Devuelve el hash hexadecimal para guardarlo en los metadatos.
    return digest.hexdigest()


def _output_path(
    output_root: Path,
    stage: str,
    phase: str,
    name: str,
    smoke: bool,
) -> tuple[Path, Path, Path]:
    """Forma rutas de resultado final, checkpoint reanudable y registro de errores."""
    # Smoke runs quedan aisladas para no mezclarse con los resultados definitivos.
    root = output_root / "smoke" if smoke else output_root
    # Separa archivos por experimento y fase (por ejemplo q4/development).
    directory = root / stage / phase
    # JSON final solo aparece cuando todas las preguntas de la celda terminaron.
    final = directory / f"{name}.json"
    # JSONL se actualiza pregunta por pregunta para permitir reanudar.
    checkpoint = directory / f"{name}.jsonl"
    # Los errores quedan aparte de respuestas válidas.
    errors = directory / f"{name}.errors.jsonl"
    # Devuelve las tres ubicaciones para que el generador las use.
    return final, checkpoint, errors


def _generate_condition(
    records: list[dict[str, Any]],
    contexts_for_record: Callable[
        [dict[str, Any]], tuple[list[str], dict[str, Any]]
    ],
    condition: str,
    budget: int,
    stage: str,
    phase: str,
    name: str,
    output_root: Path,
    runs_root: Path,
    client: LLMClient | None,
    encoding: tiktoken.Encoding,
    dry_run: bool,
    smoke: bool,
    model: str,
) -> tuple[int, int]:
    """Genera una condición, reanuda checkpoints compatibles y escribe salida ordenada.

    La huella vincula predicciones con prompt, muestra, protocolo y código generador.
    Los JSONL incompletos permiten continuar sin repetir preguntas ya guardadas.
    """
    # Obtiene destinos del JSON final, checkpoint incremental y archivo de errores.
    final_path, checkpoint_path, errors_path = _output_path(
        output_root, stage, phase, name, smoke
    )
    # Conserva el orden esperado de IDs; se usará para validar o ordenar al final.
    expected_ids = [str(record["id"]) for record in records]
    # La configuración captura factores que pueden cambiar la respuesta experimental.
    config = {
        # Versión manual del protocolo de Q3/Q4.
        "protocol_version": PROTOCOL_VERSION,  # Versión del protocolo experimental.
        # Etapa, fase, condición y presupuesto de esta celda.
        "experiment": stage.upper(),  # Q3 o Q4.
        "phase": phase,  # all para Q3, development/confirmation para Q4.
        "condition": condition,  # Variante o método de recuperación.
        "budget": budget,  # Presupuesto solicitado para contexto.
        "model": model,  # Modelo utilizado para generar respuestas.
        "temperature": GENERATION_TEMPERATURE,  # Control de aleatoriedad.
        "max_output_tokens": MAX_OUTPUT_TOKENS,  # Límite de tokens de respuesta.
        # Hash del prompt: detecta cambios de instrucciones que alteran las generaciones.
        "generation_prompt_sha256": hashlib.sha256(
            (GENERATION_SYSTEM + "\n" + GENERATION_USER).encode()
        ).hexdigest(),
        # Huella del archivo ejecutado: invalida checkpoints al cambiar este script.
        "generator_script_sha256": _file_sha256(Path(__file__)),
        # Huella de los IDs/muestra base y de los IDs usados en esta celda.
        "sample_file_sha256": _file_sha256(  # Detecta cambios en la muestra congelada.
            ROOT / "research" / "configs" / "questions_400.json"
        ),
        "question_ids_sha256": hashlib.sha256("\n".join(expected_ids).encode()).hexdigest(),  # IDs exactos de esta celda.
    }
    # Q4 también depende del archivo que fija desarrollo frente a confirmación.
    if stage == "q4":
        config["q4_split_sha256"] = _file_sha256(Q4_SPLIT_PATH)
    # Huella compacta usada para comparar configuración en cada predicción guardada.
    fingerprint = _config_fingerprint(config)

    # Si hay salida final, se permite omitirla solo cuando IDs y configuración coinciden.
    if final_path.exists():
        # Lee la salida existente y extrae su secuencia de IDs.
        existing = json.loads(final_path.read_text(encoding="utf-8"))
        existing_ids = [str(row.get("id")) for row in existing]
        # Una salida completa e idéntica no necesita otra llamada al modelo.
        if existing_ids == expected_ids and all(
            row.get("experiment_metadata", {}).get("config_fingerprint") == fingerprint
            for row in existing
        ):
            print(f"SKIP complete: {final_path}", flush=True)
            return len(records), 0
        # Evita sobrescribir silenciosamente resultados incompatibles o incompletos.
        raise FileExistsError(
            f"Refusing to overwrite incompatible output {final_path}; choose a new --output-root"
        )

    # Dry-run valida recuperación, tokens y metadatos sin llamar al LLM ni persistir.
    if dry_run:
        # Recorre todas las preguntas seleccionadas para revisar el flujo completo.
        for index, record in enumerate(records, start=1):
            # La función recibida construye el contexto específico de esta pregunta.
            contexts, metadata = contexts_for_record(record)
            # Calcula longitud una vez serializado como será en el prompt.
            actual = _rendered_tokens(contexts, encoding)
            # El presupuesto configurado es un límite duro.
            if actual > budget:
                raise ValueError(
                    f"{record['id']}: rendered context {actual} exceeds budget {budget}"
                )
            # En Q3 las variantes de grafo deben tener el conteo emparejado exacto.
            if stage == "q3" and condition != "rag_reference":
                if actual != metadata["matched_context_tokens"]:
                    raise ValueError(f"{record['id']}: Q3 matched-token assertion failed")
            # Muestra solo primera y última pregunta para dar una señal de avance legible.
            if index == 1 or index == len(records):
                print(
                    f"DRY-RUN {name}: {index}/{len(records)} "
                    f"tokens={actual} metadata={metadata}",
                    flush=True,
                )
        # Indica cuántos registros se verificaron y que no hubo errores de dry-run.
        return len(records), 0

    # Recupera todas las líneas previamente guardadas para esta condición.
    saved = read_jsonl(checkpoint_path)
    # Mapa ID -> predicción permite saltar lo ya hecho y reconstruir orden final.
    saved_by_id: dict[str, dict[str, Any]] = {}
    # Comprueba que cada checkpoint pertenezca a esta muestra y configuración.
    for row in saved:
        record_id = str(row.get("id", ""))
        if record_id not in expected_ids:
            raise ValueError(f"Unexpected checkpoint ID {record_id!r} in {checkpoint_path}")
        if row.get("experiment_metadata", {}).get("config_fingerprint") != fingerprint:
            raise ValueError(f"Checkpoint config mismatch in {checkpoint_path}")
        # Si una línea se repitió, la última versión queda asociada al ID.
        saved_by_id[record_id] = row
    # El set hace rápidas las comprobaciones de IDs ya procesados.
    done = set(saved_by_id)
    # Contador de errores nuevos en esta ejecución; el reloj mide solo este reintento.
    errors = 0
    started = time.monotonic()
    # Asegura que exista el destino de los manifiestos.
    runs_root.mkdir(parents=True, exist_ok=True)

    # Recorre la muestra en su orden congelado y procesa solo IDs pendientes.
    for index, record in enumerate(records, start=1):
        # ID estable utilizado para asociar salida y checkpoint.
        record_id = str(record["id"])
        # Saltar un ID guardado hace seguro reanudar después de una interrupción.
        if record_id in done:
            continue
        try:
            # Construye los contextos con la función de la condición actual.
            contexts, metadata = contexts_for_record(record)
            # Comprueba antes de facturar/calcular una generación que el presupuesto se cumple.
            actual_tokens = _rendered_tokens(contexts, encoding)
            if actual_tokens > budget:
                raise ValueError(
                    f"Rendered context {actual_tokens} exceeds budget {budget}"
                )
            # Vuelve a verificar la condición central de emparejamiento Q3.
            if stage == "q3" and condition != "rag_reference":
                if actual_tokens != metadata["matched_context_tokens"]:
                    raise ValueError("Q3 contexts do not have exactly matched token counts")
            # Envía pregunta y contexto al modelo configurado para producir la respuesta.
            answer = generate_answer(str(record["question"]), contexts, client)
            # Arma el esquema común que también entiende el evaluador oficial.
            prediction = build_prediction(
                record,
                contexts,
                answer,
                condition,
                budget,
                metadata={
                    # Registra tokens y modelo junto con configuración y detalles específicos.
                    "context_tokens": actual_tokens,
                    "model": model,
                    "experiment_metadata": {
                        **config,
                        **metadata,
                        "config_fingerprint": fingerprint,
                    },
                },
            )
            # Guarda inmediatamente la predicción para minimizar trabajo perdido si falla.
            append_jsonl(checkpoint_path, prediction)
            # El manifiesto registra el estado y la huella de esta generación exitosa.
            append_jsonl(
                runs_root / "manifest.jsonl",
                {
                    "status": "ok",
                    "id": record_id,
                    "stage": stage,
                    "phase": phase,
                    "condition": condition,
                    "context_tokens": actual_tokens,
                    "output": str(final_path),
                    "config_fingerprint": fingerprint,
                },
            )
            # Actualiza memoria para no procesar este ID dos veces en la ejecución.
            saved_by_id[record_id] = prediction
            done.add(record_id)
        except Exception as error:
            # Un fallo de pregunta se registra y no detiene las demás preguntas de la celda.
            errors += 1
            # Conserva tipo y mensaje del error para depuración.
            error_record = {
                "id": record_id,
                "stage": stage,
                "phase": phase,
                "condition": condition,
                "error": f"{type(error).__name__}: {error}",
            }
            # Escribe el detalle en el log de errores y también en el manifiesto.
            append_jsonl(errors_path, error_record)
            append_jsonl(runs_root / "manifest.jsonl", {"status": "error", **error_record})
            print(f"ERROR {name} {record_id}: {error_record['error']}", flush=True)
        # La cantidad procesada cuenta éxitos, no errores.
        processed = len(done)
        # Imprime progreso cada cinco respuestas o al llegar al último registro.
        if processed and (processed % 5 == 0 or index == len(records)):
            # Calcula minutos transcurridos desde el comienzo de esta ejecución.
            elapsed = time.monotonic() - started
            print(
                f"[{name}] {processed}/{len(records)} saved "
                f"| errors={errors} | elapsed={elapsed / 60:.1f} min",
                flush=True,
            )

    # El archivo final solo se crea cuando todos los IDs esperados tienen predicción.
    if len(done) == len(records):
        # Ordena resultados para que coincidan con el orden original de la muestra.
        ordered = [saved_by_id[record_id] for record_id in expected_ids]
        # Primero escribe un temporal para no dejar JSON final a medio escribir.
        temporary = final_path.with_suffix(".json.tmp")
        # Crea la carpeta de la condición si todavía no existe.
        final_path.parent.mkdir(parents=True, exist_ok=True)
        # allow_nan=False garantiza que el archivo sea JSON válido estricto.
        temporary.write_text(
            json.dumps(ordered, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        # El reemplazo atómico evita presentar un archivo incompleto como completo.
        temporary.replace(final_path)
        # Informa la ruta de la celda finalizada.
        print(f"COMPLETE {name}: {final_path}", flush=True)
    else:
        # Informa el número guardado y la ubicación desde la que se podrá reanudar.
        print(
            f"INCOMPLETE {name}: {len(done)}/{len(records)} saved; "
            f"rerun this command to resume from {checkpoint_path}",
            flush=True,
        )
    # Devuelve número de respuestas y errores a main para sumar el proceso total.
    return len(done), errors


def _records_for_ids(
    records: list[dict[str, Any]],
    ids: list[str],
    limit_per_cell: int | None,
) -> list[dict[str, Any]]:
    """Filtra IDs solicitados y, si se pide, limita cada celda corpus/tipo."""
    # Intersecta la muestra original con la lista de IDs del split elegido.
    selected = [record for record in records if str(record["id"]) in set(ids)]
    # El orden fijo evita diferencias por el orden de entrada.
    selected.sort(key=lambda row: str(row["id"]))
    # Sin límite se procesan todos los registros seleccionados.
    if limit_per_cell is None:
        return selected
    # Agrupa por corpus y tipo para que el smoke test no quede sesgado a un solo grupo.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    # El prefijo del ID indica a cuál corpus pertenece la pregunta.
    for record in selected:
        corpus = "novel" if str(record["id"]).startswith("Novel") else "medical"
        grouped[(corpus, str(record["question_type"]))].append(record)
    # Recorre estratos en orden fijo y toma hasta N preguntas por cada uno.
    limited = []
    for corpus in ("novel", "medical"):
        for level in LEVELS:
            limited.extend(grouped[(corpus, level)][:limit_per_cell])
    # Devuelve la muestra reducida usada exclusivamente para dry-runs/smoke tests.
    return limited


def main() -> int:
    """Lee opciones, prepara recursos compartidos y ejecuta las etapas solicitadas."""
    # Declara el programa de línea de comandos.
    parser = argparse.ArgumentParser(
        description="Generate Q3 controlled variants and the Q4 depth grid."
    )
    # Permite ejecutar Q3, Q4 o ambas etapas.
    parser.add_argument("--stages", nargs="+", choices=("q3", "q4"), default=("q3", "q4"))
    # Q4 puede correr en desarrollo o, tras escoger candidato, en confirmación.
    parser.add_argument("--q4-phase", choices=("development", "confirmation"), default="development")
    # La confirmación requiere exactamente un método y una profundidad elegida.
    parser.add_argument("--confirm-method", choices=Q4_METHODS)
    parser.add_argument("--confirm-depth", type=int, choices=Q4_DEPTHS)
    # Permite cambiar modelo o endpoint local desde terminal.
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    # En desarrollo se pueden seleccionar algunas profundidades; por defecto 1 a 6.
    parser.add_argument("--q4-depths", nargs="+", type=int, choices=Q4_DEPTHS, default=Q4_DEPTHS)
    # Rutas personalizadas para no sobrescribir resultados guardados en ubicaciones estándar.
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    # Esta opción construye contextos y revisa tokens sin invocar el modelo.
    parser.add_argument("--dry-run", action="store_true")
    # Reduce tamaño de cada estrato y dirige las salidas a una subcarpeta smoke/.
    parser.add_argument(
        "--limit-per-cell",
        type=int,
        help="Limit to N records per corpus/question type; output is isolated under smoke/.",
    )
    # Convierte texto de terminal a variables de Python.
    args = parser.parse_args()

    # Rechaza un límite que no permita seleccionar ninguna fila.
    if args.limit_per_cell is not None and args.limit_per_cell <= 0:
        parser.error("--limit-per-cell must be positive")
    # Una confirmación no tiene sentido sin un candidato único preseleccionado.
    if args.q4_phase == "confirmation":
        if args.confirm_method is None or args.confirm_depth is None:
            parser.error(
                "confirmation requires --confirm-method and --confirm-depth"
            )
    # Evita que parámetros de confirmación se mezclen por accidente con desarrollo.
    elif args.confirm_method is not None or args.confirm_depth is not None:
        parser.error("--confirm-method/--confirm-depth are only valid for confirmation")
    # Una ejecución sin etapa no tendría trabajo que validar/generar.
    if args.dry_run and not args.stages:
        parser.error("at least one stage is required")

    # Lee los 400 registros con IDs, corpus y tipo congelados.
    sample = load_sample()
    records = list(sample["records"])
    # Obtiene la partición Q4, verificando o creando sus IDs deterministas.
    split = _load_or_create_q4_split(records)
    # En desarrollo usa 320; en confirmación usa los 80 reservados.
    q4_ids = split[
        "development_ids" if args.q4_phase == "development" else "confirmation_ids"
    ]
    # Selecciona registros Q4 y aplica límite opcional para pruebas de humo.
    q4_records = _records_for_ids(records, q4_ids, args.limit_per_cell)
    # Q3 usa todas las preguntas; el límite solo reduce pruebas pequeñas.
    q3_records = _records_for_ids(records, [str(row["id"]) for row in records], args.limit_per_cell)
    # Parámetros de modelo: lo escrito en CLI prevalece sobre defaults de Ollama.
    model = args.model or DEFAULT_OLLAMA_MODEL
    base_url = args.base_url or DEFAULT_OLLAMA_BASE_URL
    # En dry-run no se inicializa cliente para asegurar que no haya llamada al modelo.
    client = None
    # Para generación real se configura Ollama con temperatura y límite de salida fijos.
    if not args.dry_run:
        client = LLMClient(
            model=model,
            temperature=GENERATION_TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
            base_url=base_url,
            provider="ollama",
        )
    # Carga una vez el tokenizador usado por presupuesto y emparejamiento.
    encoding = tiktoken.get_encoding(TOKENIZER_NAME)
    # Imprime configuración de alto nivel para registrar cómo se lanzó el proceso.
    print(
        f"Q3/Q4 generation | model={model} | temperature={GENERATION_TEMPERATURE} "
        f"| tokenizer={TOKENIZER_NAME} | budget={Q3_BUDGET} | dry_run={args.dry_run}",
        flush=True,
    )
    # Informa cantidad de preguntas de cada etapa para detectar selección inesperada.
    print(
        f"Selected records: Q3={len(q3_records)}; "
        f"Q4/{args.q4_phase}={len(q4_records)}",
        flush=True,
    )

    # Cachés en memoria evitan volver a cargar corpus, grafos, índices y hashes.
    resources_by_split: dict[str, ExperimentResources] = {}
    indexes_by_source: dict[tuple[str, str], GraphIndex] = {}
    graph_hashes: dict[tuple[str, str], str] = {}

    def resources_for(record: dict[str, Any]) -> ExperimentResources:
        """Carga una vez los recursos del corpus al que pertenece una pregunta."""
        # El prefijo del ID selecciona el dataset y la carpeta de grafos.
        split_name = "novel" if str(record["id"]).startswith("Novel") else "medical"
        # Solo hace carga de disco la primera vez que se solicita este corpus.
        if split_name not in resources_by_split:
            print(f"Loading {split_name} corpus and graphs...", flush=True)
            # Carga textos y grafos del split desde las rutas conocidas.
            resources_by_split[split_name] = load_resources(
                split_name, graph_dir=GRAPHS / split_name
            )
            # Novel tiene 20 libros; Medical usa un único corpus agregado.
            expected_sources = 20 if split_name == "novel" else 1
            # Una carga incompleta puede invalidar toda la selección por fuente.
            if len(resources_by_split[split_name].graphs) != expected_sources:
                raise RuntimeError(
                    f"Expected {expected_sources} {split_name} graphs, found "
                    f"{len(resources_by_split[split_name].graphs)}"
                )
        # Devuelve el objeto en memoria para reutilizarlo.
        return resources_by_split[split_name]

    def index_for(record: dict[str, Any]) -> GraphIndex:
        """Construye y reutiliza un índice de recuperación por fuente de grafo."""
        # Identifica split y nombre de libro/corpus para encontrar el grafo correcto.
        split_name = "novel" if str(record["id"]).startswith("Novel") else "medical"
        source = str(record["source"])
        # La clave compuesta evita colisiones de nombres entre splits.
        key = (split_name, source)
        # Construye el índice solo una vez para cada fuente.
        if key not in indexes_by_source:
            # Recupera el grafo desde los recursos ya cargados.
            graph = resources_for(record).graphs.get(source)
            # Si falta el grafo, es error de insumo, no debe sustituirse silenciosamente.
            if graph is None:
                raise KeyError(f"No graph loaded for source {source}")
            # seed_k configura cuántas entidades iniciales considera el índice.
            indexes_by_source[key] = GraphIndex(graph, seed_k=10)
        # Devuelve el índice almacenado.
        return indexes_by_source[key]

    def graph_hash_for(record: dict[str, Any]) -> str:
        """Calcula una vez y reutiliza el hash del grafo fuente de la pregunta."""
        # Determina el split y archivo JSON del grafo.
        split_name = "novel" if str(record["id"]).startswith("Novel") else "medical"
        source = str(record["source"])
        key = (split_name, source)
        # Los hashes se cachean porque se incluirán en cada predicción de esa fuente.
        if key not in graph_hashes:
            graph_path = GRAPHS / split_name / f"{source}.json"
            graph_hashes[key] = _file_sha256(graph_path)
        # El digest permite detectar cambios posteriores del grafo.
        return graph_hashes[key]

    # Contadores globales que resumen todos los jobs ejecutados por esta llamada.
    total_errors = 0
    total_saved = 0
    # finally cerrará el cliente incluso si una condición falla.
    try:
        # La etapa Q3 compara diferentes transformaciones sobre el mismo subgrafo.
        if "q3" in args.stages:
            # Reporta el tamaño de la matriz condición x pregunta.
            print(
                f"=== Q3: {len(Q3_CONDITIONS)} conditions x {len(q3_records)} questions "
                f"at {Q3_BUDGET} tokens ===",
                flush=True,
            )
            # Los cinco contextos de grafo se construyen una vez por pregunta y se cachean.
            q3_variants_by_id: dict[
                str, dict[str, tuple[list[str], dict[str, Any]]]
            ] = {}
            # Genera por condición para conservar archivos independientes y reanudables.
            for condition in Q3_CONDITIONS:
                # Esta función adapta la condición seleccionada al generador común.
                def q3_context(record: dict[str, Any], selected=condition):
                    # RAG usa retrieval textual y no participa en el grupo de grafos emparejados.
                    if selected == "rag_reference":
                        return _q3_rag_context(
                            record, resources_for(record), encoding
                        )
                    # Los contextos de grafo se calculan y guardan la primera vez por ID.
                    record_id = str(record["id"])
                    if record_id not in q3_variants_by_id:
                        q3_variants_by_id[record_id] = _q3_graph_variants(
                            index_for(record),
                            str(record["question"]),
                            Q3_BUDGET,
                            record_id,
                            encoding,
                        )
                    # Selecciona la variante pedida de las ya construidas.
                    variants = q3_variants_by_id[record_id]
                    contexts, metadata = variants[selected]
                    # Agrega hash de procedencia del grafo al metadato de la condición.
                    return contexts, {
                        **metadata,
                        "source_graph_sha256": graph_hash_for(record),
                    }

                # Ejecuta la condición con el checkpoint y configuración compartidos.
                saved, errors = _generate_condition(
                    q3_records,
                    q3_context,
                    condition,
                    Q3_BUDGET,
                    "q3",
                    "all",
                    f"q3_{condition}_{Q3_BUDGET}",
                    args.output_root,
                    args.runs_root,
                    client,
                    encoding,
                    args.dry_run,
                    args.limit_per_cell is not None,
                    model,
                )
                # Suma el resultado de esta condición al total de ejecución.
                total_saved += saved
                total_errors += errors

        # La etapa Q4 compara algoritmos/profundidades sobre la partición elegida.
        if "q4" in args.stages:
            # Confirmación ejecuta solo el candidato previamente especificado.
            if args.q4_phase == "confirmation":
                q4_configs = [(args.confirm_method, args.confirm_depth)]
            else:
                # Desarrollo construye producto cartesiano de métodos y profundidades.
                q4_configs = [
                    (method, depth)
                    for method in Q4_METHODS
                    for depth in args.q4_depths
                ]
            # Muestra el número de celdas que se van a evaluar en esta fase.
            print(
                f"=== Q4/{args.q4_phase}: {len(q4_configs)} configurations x "
                f"{len(q4_records)} questions at {Q4_BUDGET} tokens ===",
                flush=True,
            )
            # Cada celda cambia el método o la profundidad indicados.
            for method, depth in q4_configs:
                # Captura método/profundidad actuales como valores por defecto.
                def q4_context(
                    record: dict[str, Any],
                    selected_method=method,
                    selected_depth=depth,
                ):
                    # Genera contexto de esta pregunta bajo la celda Q4 actual.
                    contexts, metadata = _q4_context(
                        index_for(record),
                        selected_method,
                        str(record["question"]),
                        selected_depth,
                        Q4_BUDGET,
                        encoding,
                    )
                    # Añade el hash del grafo que originó el contexto.
                    return contexts, {
                        **metadata,
                        "source_graph_sha256": graph_hash_for(record),
                    }

                # El nombre codifica método, profundidad y presupuesto en los artefactos.
                name = f"q4_{method}_h{depth}_{Q4_BUDGET}"
                # Reutiliza el mismo motor de generación, persistencia y reanudación.
                saved, errors = _generate_condition(
                    q4_records,
                    q4_context,
                    method,
                    Q4_BUDGET,
                    "q4",
                    args.q4_phase,
                    name,
                    args.output_root,
                    args.runs_root,
                    client,
                    encoding,
                    args.dry_run,
                    args.limit_per_cell is not None,
                    model,
                )
                # Acumula éxitos y errores de todas las celdas.
                total_saved += saved
                total_errors += errors
    finally:
        # Cierra conexiones de red cuando existe cliente inicializado.
        if client is not None and client.client is not None:
            client.client.close()

    # Informe final compacto para la terminal/log.
    print(
        f"Q3/Q4 generation finished: saved/skipped={total_saved}; "
        f"errors={total_errors}; dry_run={args.dry_run}",
        flush=True,
    )
    # Devuelve fallo si alguna pregunta produjo error; de otro modo indica éxito.
    return 1 if total_errors else 0


# Solo ejecuta el generador cuando se llama directamente desde terminal.
if __name__ == "__main__":
    # Propaga el estado de main al shell para scripts de automatización.
    raise SystemExit(main())
