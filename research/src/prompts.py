"""Plantillas de instrucciones para extracción, generación y evaluación."""

from __future__ import annotations

from .budget import format_contexts  # Numera y concatena los contextos antes de insertarlos en el prompt.

# Instrucción fija para la extracción: obliga a responder con triples estructurados.
EXTRACTION_SYSTEM = (
    "You extract knowledge-graph triples from text. "
    "Return only a JSON array. Each item must have exactly the keys "
    "head, relation, tail, and confidence. Use concise entity names. "
    "Do not invent facts that are not supported by the text."
)

# El mensaje de usuario aporta la fuente, el ID de fragmento y el texto real.
EXTRACTION_USER = """Corpus: {corpus_name}
Chunk id: {chunk_id}
Text:
{chunk_text}

Extract all factual triples supported by this chunk.
Return JSON only, for example:
[{{"head": "entity", "relation": "relation", "tail": "entity", "confidence": 0.9}}]
"""

# Instrucción del LLM que redacta la respuesta usando únicamente la evidencia.
GENERATION_SYSTEM = (
    "Answer the question using only the provided context. "
    "If the context is insufficient, say so explicitly. "
    "Be concise for factual questions and complete for summary or creative tasks."
)

# Formato de entrada de generación; los valores se insertan con .format().
GENERATION_USER = """Context:
{context}

Question: {question}

Answer:"""

# Instrucción del evaluador para comparar generación, referencia y evidencia.
JUDGE_SYSTEM = (
    "You are a strict evaluator. Compare the generated answer with the reference answer "
    "and the provided evidence. Return only the requested JSON."
)

# Campos que el llamador debe completar al solicitar un juicio estructurado.
JUDGE_USER = """Question: {question}
Reference answer: {reference}
Generated answer: {answer}
Evidence: {evidence}

Return JSON with the requested metric fields and a short justification.
"""


def build_extraction_prompt(corpus_name: str, chunk_id: int | str, chunk_text: str) -> str:
    """Inserta metadatos y texto del fragmento en la plantilla de extracción."""
    return EXTRACTION_USER.format(
        corpus_name=corpus_name,  # Identifica el corpus consultado.
        chunk_id=chunk_id,  # Permite rastrear el origen del triple generado.
        chunk_text=chunk_text,  # Texto del que deben salir exclusivamente las relaciones.
    )


def build_generation_prompt(question: str, contexts: list[str]) -> str:
    """Inserta pregunta y contexto numerado en el prompt de respuesta."""
    return GENERATION_USER.format(context=format_contexts(contexts), question=question)  # Rellena los dos espacios de la plantilla.


def build_judge_prompt(question: str, reference: str, answer: str, evidence: str) -> str:
    """Construye el texto con el que un juez evalúa la respuesta generada."""
    return JUDGE_USER.format(
        question=question,  # Pregunta a la que debía responder el modelo.
        reference=reference,  # Respuesta de referencia del dataset.
        answer=answer,  # Respuesta producida durante el experimento.
        evidence=evidence,  # Elementos de evidencia entregados al juez.
    )
