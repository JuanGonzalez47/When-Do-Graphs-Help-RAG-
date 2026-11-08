"""Generación de respuestas y ensamblado de registros de predicción."""

from __future__ import annotations

from typing import Any  # Los registros incluyen campos del benchmark y metadatos experimentales.

from .budget import pack_contexts  # Comparte el empaquetador de contextos con el recuperador.
from .llm import LLMClient  # Cliente local o remoto compatible con el proyecto.
from .prompts import GENERATION_SYSTEM, build_generation_prompt  # Plantilla y rol de generación.


def generate_answer(
    question: str,
    contexts: list[str],
    client: LLMClient,
    prompt_contexts: list[str] | None = None,
) -> str:
    """Pide una respuesta al modelo usando los contextos seleccionados."""
    prompt = build_generation_prompt(question, prompt_contexts or contexts)  # Usa la versión formateada si fue preparada.
    return client.complete(GENERATION_SYSTEM, prompt).strip()  # Envía el prompt y quita espacios sobrantes.


def build_prediction(
    record: dict[str, Any],
    contexts: list[str],
    generated_answer: str,
    condition: str,
    budget: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combina pregunta, contexto, respuesta y etiquetas en el esquema de evaluación."""
    # Los dos datasets oficiales usan campos de evidencia con nombres distintos.
    evidence = record.get("evidence", record.get("evidence_triple", record.get("evidence_relations", "")))
    prediction = {
        "id": record["id"],  # Clave usada para emparejar la salida con la pregunta oficial.
        "source": record.get("source"),  # Documento o fuente de origen, si existe.
        "question": record["question"],  # Pregunta enviada al modelo.
        "question_type": record["question_type"],  # Estrato/tipo asignado por el benchmark.
        "context": contexts,  # Evidencia contextual efectivamente entregada.
        "evidence": evidence,  # Referencia de evidencia para evaluación de retrieval.
        "ground_truth": record["answer"],  # Respuesta de referencia del dataset.
        "generated_answer": generated_answer,  # Texto devuelto por el LLM.
        "condition": condition,  # Variante experimental que produjo el contexto.
        "budget": budget,  # Presupuesto solicitado para esa variante.
    }
    if metadata:
        prediction.update(metadata)  # Conserva información adicional de auditoría.
    return prediction  # Devuelve un registro listo para persistir/evaluar.


def prepare_contexts(
    contexts: list[Any],
    budget: int,
    encoding_name: str = "cl100k_base",
) -> tuple[list[str], int]:
    """Normaliza y limita los contextos a un presupuesto tokenizado."""
    return pack_contexts(contexts, max_tokens=budget, encoding_name=encoding_name)  # Devuelve textos y tokens consumidos.
