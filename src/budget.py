"""Conteo de tokens, recorte de texto y empaquetado de contexto."""

from __future__ import annotations

from typing import Any, Iterable  # Tipos flexibles para entradas y listas de contextos.

import tiktoken  # Tokenizador usado para respetar los presupuestos del experimento.


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Cuenta tokens del texto con la codificación solicitada."""
    encoding = tiktoken.get_encoding(encoding_name)  # Obtiene el codificador por su nombre.
    return len(encoding.encode(text))  # Cuenta IDs de tokens, no palabras ni caracteres.


def truncate_to_tokens(text: str, max_tokens: int, encoding_name: str = "cl100k_base") -> str:
    """Conserva el prefijo de texto que cabe en el máximo de tokens."""
    if max_tokens <= 0:
        return ""  # Un presupuesto cero/negativo no permite conservar contenido.
    encoding = tiktoken.get_encoding(encoding_name)  # Usa la misma codificación al medir y recortar.
    tokens = encoding.encode(text)  # Convierte el texto completo a IDs de token.
    if len(tokens) <= max_tokens:
        return text  # Evita reconstruirlo si ya cabe en el presupuesto.
    return encoding.decode(tokens[:max_tokens])  # Decodifica solo los primeros tokens permitidos.


def _text_of(item: Any) -> str:
    if isinstance(item, str):
        return item  # Los contextos simples ya son cadenas.
    if isinstance(item, dict):
        for key in ("text", "content", "passage", "context"):
            if key in item:
                return str(item[key])  # Acepta las claves comunes de texto recuperado.
    return str(item)  # Convierte entradas inesperadas para no perder su representación.


def pack_contexts(
    contexts: Iterable[Any],
    max_tokens: int,
    encoding_name: str = "cl100k_base",
    truncate_last: bool = True,
) -> tuple[list[str], int]:
    """Agrega contextos en orden hasta llenar el presupuesto disponible."""
    selected: list[str] = []  # Contextos que efectivamente se enviarán al modelo.
    used = 0  # Tokens acumulados en los contextos seleccionados.
    for item in contexts:
        text = _text_of(item)  # Normaliza cadenas y registros de recuperación.
        if not text:
            continue  # Ignora entradas vacías sin gastar presupuesto.
        n_tokens = count_tokens(text, encoding_name=encoding_name)  # Mide el bloque actual.
        if used + n_tokens <= max_tokens:
            selected.append(text)  # El bloque cabe completo.
            used += n_tokens  # Actualiza el espacio ya consumido.
            continue  # Prueba el siguiente bloque en el orden de entrada.
        if truncate_last and used < max_tokens:
            remaining = max_tokens - used  # Capacidad que queda para el bloque parcial.
            truncated = truncate_to_tokens(text, remaining, encoding_name=encoding_name)  # Recorta al espacio restante.
            if truncated:
                selected.append(truncated)  # Conserva el prefijo que sí cabe.
                used += count_tokens(truncated, encoding_name=encoding_name)  # Registra el tamaño real tras decodificar.
        break  # No salta bloques: conserva el orden y el prefijo de recuperación.
    return selected, used  # Devuelve contexto y conteo medido.


def format_contexts(contexts: Iterable[str], sep: str = "\n\n") -> str:
    """Une los contextos y numera cada fragmento para citarlo en el prompt."""
    return sep.join(f"[{index + 1}] {text}" for index, text in enumerate(contexts))  # La numeración empieza en uno para lectura humana.


def budget_metadata(contexts: list[str], max_tokens: int, encoding_name: str = "cl100k_base") -> dict[str, Any]:
    """Resume el consumo del presupuesto de una lista ya seleccionada."""
    used = sum(count_tokens(text, encoding_name=encoding_name) for text in contexts)  # Cuenta nuevamente el contenido exacto.
    return {
        "budget": max_tokens,  # Límite que se solicitó.
        "contexts": len(contexts),  # Cantidad de bloques enviados.
        "context_tokens": used,  # Tokens realmente presentes.
        "remaining_tokens": max_tokens - used,  # Diferencia entre límite y consumo.
    }
