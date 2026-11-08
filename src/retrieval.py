"""Recuperación léxica de fragmentos para las condiciones RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedChunk:
    """Resultado recuperado con puntuación y campos de auditoría."""

    chunk_id: int | str  # Identificador del fragmento dentro de su fuente.
    source: str  # Documento del corpus del que proviene el fragmento.
    text: str  # Texto que puede incluirse en el contexto del LLM.
    score: float  # Similitud coseno entre consulta y fragmento TF-IDF.
    metadata: dict[str, Any] = field(default_factory=dict)  # Resto de campos originales.


class TfidfRetriever:
    """Ordena fragmentos por similitud léxica TF-IDF con la pregunta."""

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int | None = 200_000,
        stop_words: str | None = "english",
        sublinear_tf: bool = True,
    ) -> None:
        # Vectoriza unigramas/bigramas y limita el vocabulario para controlar memoria.
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,  # N-gramas que representan los términos.
            max_features=max_features,  # Tope de palabras/rasgos del vocabulario.
            stop_words=stop_words,  # Palabras vacías del idioma configurado.
            sublinear_tf=sublinear_tf,  # Suaviza el peso de repeticiones frecuentes.
        )
        self.chunks: list[dict[str, Any]] = []  # Copia de los fragmentos ya ordenados.
        self.matrix = None  # Matriz TF-IDF; queda sin crear hasta fit().

    def fit(self, chunks: Iterable[dict[str, Any]]) -> "TfidfRetriever":
        """Ajusta el vocabulario TF-IDF sobre los fragmentos disponibles."""
        self.chunks = [dict(chunk) for chunk in chunks]  # Copia entradas para no alterar al llamador.
        texts = [str(chunk.get("text", "")) for chunk in self.chunks]  # Extrae el texto a vectorizar.
        if not texts:
            raise ValueError("Cannot fit on an empty chunk list")  # Una matriz vacía no sirve para buscar.
        self.matrix = self.vectorizer.fit_transform(texts)  # Aprende vocabulario y transforma cada fragmento.
        return self  # Permite encadenar TfidfRetriever().fit(...).

    def scores(self, query: str) -> np.ndarray:
        if self.matrix is None:
            raise RuntimeError("Retriever must be fitted before search")  # Evita usar una matriz inexistente.
        query_vector = self.vectorizer.transform([query])  # Usa el vocabulario fijado en fit().
        return cosine_similarity(query_vector, self.matrix).ravel()  # Una puntuación por cada fragmento.

    def search(
        self,
        query: str,
        top_k: int = 10,
        source: str | None = None,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Devuelve hasta top_k fragmentos ordenados por similitud descendente."""
        scores = self.scores(query).copy()  # Copia para filtrar sin alterar la matriz o futuros resultados.
        if source is not None:
            mask = np.array([chunk.get("source") == source for chunk in self.chunks])  # Marca fuentes permitidas.
            scores = np.where(mask, scores, -1.0)  # Excluye otras fuentes sin cambiar el orden de datos.
        order = np.argsort(-scores, kind="stable")  # Orden descendente con desempate determinista por índice.
        results: list[RetrievedChunk] = []  # Salidas con el formato común del recuperador.
        for index in order:
            if len(results) >= top_k:
                break  # Detiene la búsqueda al alcanzar el máximo solicitado.
            score = float(scores[index])  # Convierte el escalar NumPy a float nativo.
            if score < min_score:
                break  # Al estar ordenado, los siguientes tampoco superan el mínimo.
            chunk = self.chunks[int(index)]  # Recupera el registro que corresponde a la posición.
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.get("chunk_id", int(index)),  # Usa ID guardado o índice como fallback.
                    source=str(chunk.get("source", "")),  # Conserva la fuente para filtrar/auditar.
                    text=str(chunk.get("text", "")),  # Devuelve el fragmento para construir contexto.
                    score=score,  # Guarda el ranking calculado.
                    metadata={
                        key: value
                        for key, value in chunk.items()
                        if key not in {"text", "chunk_id", "source"}  # El resto son metadatos auxiliares.
                    },
                )
            )
        return results  # Conserva el orden de mayor a menor relevancia.


def retrieved_to_contexts(chunks: Iterable[RetrievedChunk]) -> list[str]:
    """Convierte resultados tipados a la lista de cadenas usada por generación."""
    return [chunk.text for chunk in chunks]  # El orden de relevancia se mantiene.
