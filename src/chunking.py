"""División de textos del corpus en fragmentos medidos por tokens."""

from __future__ import annotations

from typing import Any  # Los fragmentos se representan como diccionarios de campos variados.

import tiktoken  # Convierte el texto en unidades iguales a las del presupuesto del LLM.


def get_encoding(name: str = "cl100k_base") -> tiktoken.Encoding:
    """Obtiene el tokenizador cuyo nombre se configura en el experimento."""
    return tiktoken.get_encoding(name)  # Usa las codificaciones conocidas por tiktoken.


def split_text_tokens(
    text: str,
    chunk_tokens: int = 256,
    chunk_overlap: int = 32,
    encoding_name: str = "cl100k_base",
) -> list[dict[str, Any]]:
    """Corta un texto en ventanas solapadas y guarda sus posiciones en tokens."""
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")  # Evita ventanas vacías o sin avance.
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")  # El solapamiento no puede retroceder.
    if chunk_overlap >= chunk_tokens:
        raise ValueError("chunk_overlap must be smaller than chunk_tokens")  # Garantiza que cada paso avance.

    encoding = get_encoding(encoding_name)  # Un mismo codificador mide y reconstruye todos los fragmentos.
    tokens = encoding.encode(text)  # Los límites se calculan sobre tokens, no caracteres.
    chunks: list[dict[str, Any]] = []  # Acumula los registros de fragmento resultantes.
    start = 0  # Posición inicial de la primera ventana.
    index = 0  # Número secuencial del fragmento.
    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))  # No lee más allá del final del documento.
        chunk_tokens_slice = tokens[start:end]  # Toma la ventana actual.
        chunk_text = encoding.decode(chunk_tokens_slice)  # Reconstruye el fragmento legible.
        chunks.append(
            {
                "chunk_id": index,  # ID local al texto fuente.
                "text": chunk_text,  # Texto que se extraerá o recuperará.
                "start_token": start,  # Inicio inclusivo en la secuencia tokenizada.
                "end_token": end,  # Fin exclusivo en la secuencia tokenizada.
                "n_tokens": len(chunk_tokens_slice),  # Tamaño exacto de esta ventana.
            }
        )
        if end == len(tokens):
            break  # La última ventana ya cubrió el resto del documento.
        start += chunk_tokens - chunk_overlap  # Desplaza la ventana conservando el solapamiento.
        index += 1  # Asigna el siguiente identificador.
    return chunks  # Entrega fragmentos en el orden original del texto.


def chunk_corpus(
    corpus: dict[str, str],
    chunk_tokens: int = 256,
    chunk_overlap: int = 32,
    encoding_name: str = "cl100k_base",
) -> dict[str, list[dict[str, Any]]]:
    """Aplica idéntica configuración de ventanas a cada texto con nombre."""
    return {
        name: split_text_tokens(
            text,
            chunk_tokens=chunk_tokens,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding_name,
        )
        for name, text in corpus.items()  # La clave conserva el nombre de la fuente del corpus.
    }


def chunk_texts(chunks: list[dict[str, Any]]) -> list[str]:
    """Extrae únicamente el contenido textual de los registros de fragmento."""
    return [str(chunk["text"]) for chunk in chunks]  # Mantiene el orden de entrada.
