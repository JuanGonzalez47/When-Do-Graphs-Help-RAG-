"""Orquestador de condiciones que genera predicciones en formato del benchmark."""

from __future__ import annotations

import argparse  # Parsea opciones del usuario.
import csv  # Escribe el manifiesto tabular de celdas ejecutadas.
import time  # Mide el tiempo de generación.
from pathlib import Path  # Construye rutas a predicciones/checkpoints.
from typing import Any, Iterable  # Tipos flexibles de registros y condiciones.

from .budget import count_tokens  # Cuenta tokens de salida.
from .conditions import ALL_CONDITIONS, Q1_CONDITIONS, Q2_CONDITIONS, ExperimentResources, build_contexts, load_resources  # Define contexto de cada variante.
from .data import LEVELS, load_sample, sample_records  # Muestra fija y filtros por tipo.
from .generation import build_prediction, generate_answer  # Construye respuestas y registros oficiales.
from .io import append_jsonl, read_jsonl, write_json  # Checkpoints incrementales y archivo final.
from .llm import LLMClient  # Proveedor de generación configurado para la ejecución.
from .paths import GRAPHS, OUTPUTS, RESULTS  # Ubicaciones comunes de grafos y resultados.

DEFAULT_OUTPUTS = OUTPUTS / "q1_q2" / "predictions"  # JSON final por condición y corpus.
DEFAULT_RAW = OUTPUTS / "q1_q2" / "raw"  # JSONL por pregunta para reanudar fallos/interrupciones.
DEFAULT_MANIFEST = RESULTS / "manifest.csv"  # Resumen de duración, cantidad y archivos.

OFFICIAL_FIELDS = [  # Campos que requiere el evaluador del repositorio clonado.
    "id",
    "question",
    "source",
    "context",
    "evidence",
    "question_type",
    "generated_answer",
    "ground_truth",
    "condition",
    "budget",
]


def output_path(output_dir: str | Path, split: str, condition: str, budget: int) -> Path:
    """Construye la ruta de predicciones completas de una celda."""
    return Path(output_dir) / split / f"{condition}_{budget}.json"  # Una celda es corpus × condición × presupuesto.


def raw_path(raw_dir: str | Path, split: str, condition: str, budget: int) -> Path:
    """Construye la ruta del checkpoint de respuestas individuales."""
    return Path(raw_dir) / split / f"{condition}_{budget}.jsonl"  # JSONL permite append y reanudación.


def _append_manifest(path: str | Path, row: dict[str, Any]) -> None:
    """Añade una fila de resumen al CSV sin repetir el encabezado."""
    path = Path(path)  # Normaliza la ruta entregada.
    path.parent.mkdir(parents=True, exist_ok=True)  # Prepara directorio del manifiesto.
    exists = path.exists()  # Determina si hay que escribir el encabezado.
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))  # Usa claves de fila como columnas.
        if not exists:
            writer.writeheader()  # Inicializa una tabla CSV nueva.
        writer.writerow(row)  # Añade el resumen de esta celda.


def run_condition(
    records: list[dict[str, Any]],
    condition: str,
    budget: int,
    resources: ExperimentResources,
    split: str,
    client: LLMClient | None = None,
    output_dir: str | Path = DEFAULT_OUTPUTS,
    raw_dir: str | Path = DEFAULT_RAW,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    limit: int | None = None,
    dry_run: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Genera una celda pregunta por pregunta y permite reanudar desde JSONL."""
    if condition not in ALL_CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")  # Falla antes de producir datos para una variante inválida.
    if limit is not None:
        records = records[:limit]  # Reduce a un prefijo para smoke tests.
    raw = raw_path(raw_dir, split, condition, budget)  # Ruta del checkpoint por condición.
    done: set[str] = set()  # IDs que ya tienen una predicción guardada.
    predictions: list[dict[str, Any]] = []  # Salidas completas de esta celda.
    if resume and raw.exists():
        predictions = read_jsonl(raw)  # Lee respuestas ya generadas en corridas anteriores.
        done = {str(prediction["id"]) for prediction in predictions}  # Evita repetir preguntas completadas.
    elif raw.exists():
        raw.unlink()  # Sin --resume, comienza de nuevo y descarta el checkpoint de esa celda.
    started = time.time()  # Inicio usado para calcular duración.
    errors = 0  # Errores ocurridos en esta invocación.
    processed = len(done)  # Incluye predicciones recuperadas del checkpoint.
    for record in records:
        record_id = str(record["id"])  # Clave estable de la pregunta.
        if record_id in done:
            continue  # La pregunta ya fue generada en una ejecución anterior.
        try:
            contexts, metadata = build_contexts(record, condition, budget, resources)  # Recupera texto/grafo y mide tokens.
            if dry_run or client is None:
                answer = contexts[0] if contexts else ""  # Dry-run no llama al modelo; produce texto de prueba.
            else:
                answer = generate_answer(str(record["question"]), contexts, client)  # Genera respuesta sobre los contextos.
            prediction = build_prediction(
                record,  # Campos originales de la pregunta.
                contexts,  # Contexto realmente enviado.
                answer,  # Respuesta generada o texto de prueba.
                condition,  # Etiqueta experimental.
                budget,  # Límite configurado.
                metadata={"condition_metadata": metadata},  # Trazabilidad de selección.
            )
            prediction["context_tokens"] = metadata.get("context_tokens", 0)  # Registra consumo efectivo de entrada.
            prediction["answer_tokens"] = count_tokens(answer)  # Mide longitud de la respuesta generada.
            if client is not None:
                prediction["model"] = client.model  # Identifica modelo asociado a la respuesta real.
            predictions.append(prediction)  # Conserva registro para el JSON final.
            append_jsonl(raw, prediction)  # Persiste inmediatamente para poder reanudar.
            processed += 1  # Cuenta esta pregunta como completada.
            if processed % 25 == 0 or processed == len(records):
                elapsed = time.time() - started  # Tiempo desde el inicio de esta celda.
                print(
                    f"[{split}/{condition}/{budget}] {processed}/{len(records)} "
                    f"| errors={errors} | {processed / elapsed if elapsed else 0:.2f} records/s",
                    flush=True,
                )
        except Exception as error:
            errors += 1  # Cuenta fallo sin ocultar el error en un archivo.
            append_jsonl(
                Path(raw_dir) / split / "errors.jsonl",
                {"id": record_id, "condition": condition, "budget": budget, "error": repr(error)},  # Deja contexto para depuración.
            )
    output = output_path(output_dir, split, condition, budget)  # Ruta del JSON que leerá el evaluador.
    write_json(output, predictions)  # Persiste incluso si hubo preguntas fallidas.
    elapsed = time.time() - started  # Duración total de esta llamada.
    summary = {
        "split": split,  # Corpus que se procesó.
        "condition": condition,  # Método/ablación ejecutado.
        "budget": budget,  # Presupuesto solicitado.
        "records": len(predictions),  # Cantidad de respuestas disponibles.
        "errors": errors,  # Fallos ocurridos en esta invocación.
        "context_tokens": sum(int(row.get("context_tokens", 0)) for row in predictions),  # Tokens acumulados en contextos.
        "elapsed_seconds": round(elapsed, 2),  # Tiempo de pared redondeado.
        "path": str(output),  # Ubicación útil para revisar o evaluar.
    }
    _append_manifest(manifest_path, summary)  # Agrega la fila del experimento al CSV global.
    return summary  # Permite a run_matrix imprimir/resumir las celdas.


def run_matrix(
    split: str,
    conditions: Iterable[str],
    budgets: Iterable[int],
    resources: ExperimentResources,
    client: LLMClient | None = None,
    records: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Ejecuta cada combinación de condición/presupuesto para un corpus."""
    records = records or sample_records()  # Usa muestra congelada si no se entrega otra lista.
    records = [record for record in records if _split_of(record) == split]  # Excluye preguntas del otro corpus.
    summaries = []  # Resumen por celda.
    for condition in conditions:
        for budget in budgets:
            summaries.append(
                run_condition(
                    records,
                    condition,  # Variante de recuperación a evaluar.
                    budget,  # Límite de contexto.
                    resources,  # Recursos compartidos del corpus.
                    split,  # Split que define grafos y nombres de salida.
                    client=client,  # Cliente local/remoto o None para dry-run.
                    **kwargs,  # Rutas, límite, reanudación y demás opciones del llamador.
                )
            )
    return summaries  # Lista de métricas operativas por condición y presupuesto.


def _split_of(record: dict[str, Any]) -> str:
    """Infiere el corpus desde el prefijo de ID fijado por el benchmark."""
    return "novel" if str(record["id"]).startswith("Novel") else "medical"  # Medical es el caso restante conocido.


def parse_conditions(value: str) -> list[str]:
    if value == "q1":
        return list(Q1_CONDITIONS)  # Expande alias a las cuatro condiciones definidas para Q1.
    if value == "q2":
        return list(Q2_CONDITIONS)  # Expande alias a controles/métodos de Q2.
    if value == "all":
        return list(ALL_CONDITIONS)  # Devuelve todas las variantes disponibles.
    return [item.strip() for item in value.split(",") if item.strip()]  # Admite nombres separados por coma.


def parse_budgets(value: str) -> list[int]:
    """Convierte una lista separada por comas a enteros."""
    return [int(item.strip()) for item in value.split(",") if item.strip()]  # Un valor no numérico produce ValueError explícito.


def select_per_level(records: list[dict[str, Any]], per_level: int) -> list[dict[str, Any]]:
    """Toma hasta n preguntas por tipo manteniendo todos los estratos."""
    grouped: dict[str, list[dict[str, Any]]] = {level: [] for level in LEVELS}  # Prepara un bucket por tipo oficial.
    for record in records:
        level = str(record["question_type"])  # Lee el estrato de cada registro.
        if level in grouped:
            grouped[level].append(record)  # Guarda pregunta solo si su tipo es reconocido.
    selected: list[dict[str, Any]] = []  # Muestra reducida para pruebas por estrato.
    for level in LEVELS:
        selected.extend(grouped[level][:per_level])  # Limita cada grupo de forma independiente.
    return selected  # Devuelve como máximo per_level × número de tipos.


def main() -> None:
    """Prepara recursos, cliente, muestra y matriz experimental desde CLI."""
    parser = argparse.ArgumentParser(description="Run controlled GraphRAG experiments")  # Describe el comando de terminal.
    parser.add_argument("--split", choices=["novel", "medical"], required=True)  # Selecciona un corpus.
    parser.add_argument("--conditions", default="q1", help="q1, q2, all, or comma-separated names")  # Métodos a comparar.
    parser.add_argument("--budgets", default="1024,4096")  # Presupuestos de contexto separados por coma.
    parser.add_argument("--limit", type=int, default=None)  # Límite total de preguntas de una prueba.
    parser.add_argument("--per-level", type=int, default=None, help="Use the first N questions per level")  # Límite equilibrado.
    parser.add_argument("--model", default="gpt-4o-mini")  # Modelo LLM seleccionado.
    parser.add_argument("--temperature", type=float, default=0.0)  # Temperatura de generación.
    parser.add_argument("--max-tokens", type=int, default=512)  # Tope de tokens de salida.
    parser.add_argument("--graph-dir", default=None)  # Permite apuntar a grafos alternativos.
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUTS))  # Carpeta de JSON completos.
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW))  # Carpeta de checkpoints JSONL.
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))  # CSV de resúmenes de ejecución.
    parser.add_argument("--dry-run", action="store_true")  # No invoca el modelo, pero construye contexto.
    parser.add_argument("--resume", action="store_true")  # Reutiliza preguntas guardadas en el checkpoint.
    args = parser.parse_args()  # Valida y almacena los argumentos.

    conditions = parse_conditions(args.conditions)  # Resuelve alias q1/q2 o lista explícita.
    budgets = parse_budgets(args.budgets)  # Convierte presupuestos a números.
    graph_dir = Path(args.graph_dir) if args.graph_dir else GRAPHS / args.split  # Usa grafos del corpus salvo ruta manual.
    resources = load_resources(args.split, graph_dir=graph_dir)  # Indexa corpus y grafos antes de generar.
    client = None  # Sin cliente, la ejecución no hará llamadas LLM.
    if not args.dry_run:
        client = LLMClient(
            model=args.model,  # Modelo indicado por el usuario.
            temperature=args.temperature,  # Control de aleatoriedad.
            max_tokens=args.max_tokens,  # Límite de respuesta.
        )
    sample = load_sample()  # Lee la muestra congelada, no todas las preguntas de origen.
    records = [record for record in sample["records"] if _split_of(record) == args.split]  # Separa solo el corpus pedido.
    if args.per_level is not None:
        records = select_per_level(records, args.per_level)  # Mantiene cobertura de tipos en smoke tests.
    summaries = run_matrix(
        args.split,  # Corpus objetivo.
        conditions,  # Condiciones resueltas.
        budgets,  # Presupuestos numéricos.
        resources,  # Recursos cargados para el corpus.
        client=client,  # Cliente real o None.
        records=records,  # Muestra filtrada y, opcionalmente, limitada.
        output_dir=args.output_dir,  # Destino de JSON finales.
        raw_dir=args.raw_dir,  # Destino de JSONL incrementales.
        manifest_path=args.manifest,  # Destino del manifiesto CSV.
        limit=args.limit,  # Máximo adicional de preguntas.
        dry_run=args.dry_run,  # Pasa modo sin inferencia.
        resume=args.resume,  # Pasa política de reanudación.
    )
    print(f"Completed {len(summaries)} condition-budget cells")  # Número de celdas solicitadas y procesadas.


if __name__ == "__main__":
    main()
