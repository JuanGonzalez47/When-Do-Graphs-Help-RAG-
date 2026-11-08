"""Genera una muestra balanceada y reproducible desde las preguntas oficiales."""

import argparse  # Define opciones para elegir fuentes, tamaño y semilla.
import collections  # Cuenta IDs y agrupa candidatos por fuente.
import hashlib  # Calcula la huella de los IDs seleccionados.
import json  # Lee datos oficiales y guarda la muestra/reportes.
import random  # Aporta selecciones reproducibles cuando se fija una semilla.
from pathlib import Path  # Construye rutas a archivos de preguntas y salidas.

SPLITS = {  # Nombre lógico del corpus -> archivo oficial de preguntas.
    "novel": "novel_questions.json",
    "medical": "medical_questions.json",
}

LEVELS = [  # Orden estable de los tipos para muestrear y ordenar la salida.
    "Fact Retrieval",
    "Complex Reasoning",
    "Contextual Summarize",
    "Creative Generation",
]


def load_questions(path):
    """Carga preguntas y separa los duplicados descartados para auditarlos."""
    data = json.loads(Path(path).read_text())  # Abre el JSON indicado por --questions-dir.
    seen = {}  # Guarda la primera pregunta encontrada por ID.
    dropped = []  # Registra filas que repiten un ID y no entran en la muestra.
    for q in data:
        qid = q["id"]  # Usa la identidad oficial de la pregunta.
        if qid in seen:
            dropped.append(
                {
                    "id": qid,  # ID duplicado.
                    "kept_source": seen[qid]["source"],  # Fuente de la primera fila que se conserva.
                    "dropped_source": q["source"],  # Fuente de la fila repetida.
                    "dropped_answer": q.get("answer"),  # Respuesta repetida para inspección.
                }
            )
            continue
        seen[qid] = q  # Conserva la primera copia en orden de archivo.
    return list(seen.values()), dropped  # Preguntas únicas y registro de descartes.


def sample_level(questions, n, sources, rng):
    """Elige n preguntas alternando entre fuentes dentro de un tipo."""
    by_source = collections.defaultdict(list)  # Candidatos agrupados por documento fuente.
    for q in questions:
        by_source[q["source"]].append(q)  # La clave de fuente permite repartir las selecciones.
    for src in by_source:
        by_source[src].sort(key=lambda q: q["id"])  # Fija el orden antes de aplicar la semilla.
        rng.shuffle(by_source[src])  # Mezcla de manera reproducible las preguntas de cada fuente.

    order = sorted(sources)  # Evita depender del orden interno de los sets.
    rng.shuffle(order)  # Decide qué fuente aporta primero en cada ronda.
    chosen = []  # Preguntas elegidas para este tipo.
    cursor = {src: 0 for src in order}  # Próximo índice disponible por fuente.
    while len(chosen) < n:
        progressed = False  # Sirve para detectar que no quedan candidatos.
        for src in order:
            if len(chosen) >= n:
                break  # Detiene la ronda al completar la cuota.
            bucket = by_source.get(src, [])  # Obtiene candidatos o una lista vacía.
            idx = cursor[src]  # Lee la siguiente pregunta no usada.
            if idx < len(bucket):
                chosen.append(bucket[idx])  # Toma una pregunta de esta fuente.
                cursor[src] = idx + 1  # Avanza para no repetir la misma pregunta.
                progressed = True  # Confirma que aún había datos aprovechables.
        if not progressed:
            break  # Termina si todas las fuentes agotaron sus candidatas.
    return chosen  # Puede ser menor a n; main/report debe revelar esa falta.


def main():
    """Lee opciones, genera ambos corpus y guarda muestra y reporte de balance."""
    ap = argparse.ArgumentParser()  # Construye el parser de la línea de comandos.
    ap.add_argument("--questions-dir", default="official/GraphRAG-Benchmark/Datasets/Questions")  # Directorio del clon oficial.
    ap.add_argument("--out", default="research/configs/questions_400.json")  # Archivo que recibe las preguntas elegidas.
    ap.add_argument("--report", default="research/results/sampling_report.json")  # Archivo con balance y duplicados.
    ap.add_argument("--per-level", type=int, default=50)  # Cuota deseada de preguntas por tipo y corpus.
    ap.add_argument("--seed", type=int, default=42)  # Fija aleatoriedad y permite reproducir los IDs.
    args = ap.parse_args()  # Convierte argumentos de terminal a atributos.

    rng = random.Random(args.seed)  # Comparte el generador para obtener una secuencia determinista.
    records = []  # Acumula las preguntas seleccionadas de Novel y Medical.
    report = {"seed": args.seed, "per_level": args.per_level, "splits": {}}  # Metadatos de reproducibilidad.

    for split, fname in SPLITS.items():
        questions, dropped = load_questions(Path(args.questions_dir) / fname)  # Carga y deduplica un corpus.
        sources = sorted({q["source"] for q in questions})  # Lista de fuentes disponibles para repartir la muestra.
        split_records = []  # Resultado temporal del corpus actual.
        for level in LEVELS:
            pool = [q for q in questions if q["question_type"] == level]  # No mezcla tipos de pregunta.
            picked = sample_level(pool, args.per_level, sources, rng)  # Aplica muestreo cíclico por fuente.
            split_records.extend(picked)  # Añade este tipo al resultado del corpus.
        split_records.sort(key=lambda q: (LEVELS.index(q["question_type"]), q["id"]))  # Orden canónico antes del hash.
        records.extend(split_records)  # Combina el corpus con los que ya se procesaron.

        counts = collections.Counter(q["question_type"] for q in split_records)  # Verifica cuántas preguntas quedaron por tipo.
        src_counts = collections.Counter(q["source"] for q in split_records)  # Cuenta la contribución de cada fuente.
        report["splits"][split] = {
            "available": len(questions),  # Preguntas únicas candidatas.
            "selected": len(split_records),  # Preguntas efectivamente seleccionadas.
            "by_level": dict(counts),  # Conteos por tipo para detectar cuotas incompletas.
            "by_source": dict(sorted(src_counts.items())),  # Distribución por documento.
            "dropped_duplicates": dropped,  # IDs repetidos retirados durante la carga.
        }

    by_id = collections.defaultdict(list)  # Agrupa por ID para detectar duplicados finales.
    for q in records:
        by_id[q["id"]].append(q)  # Conserva los registros asociados a cada ID.
    assert all(len(v) == 1 for v in by_id.values()), "duplicate ids in sample"  # Impide guardar una muestra duplicada.

    ids = [q["id"] for q in records]  # El orden exacto forma parte de la identidad de la muestra.
    digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()  # SHA-256 permite verificarla más tarde.
    out = {
        "seed": args.seed,  # Semilla que generó la selección.
        "per_level": args.per_level,  # Cuota por tipo de pregunta.
        "total": len(records),  # Total sumando ambos corpus.
        "sha256_ids": digest,  # Digest de los IDs ordenados.
        "question_ids": ids,  # IDs en el mismo orden que records.
        "records": records,  # Contenido de preguntas empleado en los experimentos.
    }
    report["sha256_ids"] = digest  # Vincula el reporte de balance a la misma selección.
    report["total"] = len(records)  # Total final documentado.

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)  # Prepara la carpeta del archivo de muestra.
    Path(args.out).write_text(json.dumps(out, indent=2))  # Persiste selección, IDs y digest.
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)  # Prepara la carpeta del reporte.
    Path(args.report).write_text(json.dumps(report, indent=2))  # Persiste conteos y duplicados.
    print(json.dumps({"total": len(records), "sha256_ids": digest, "report": args.report}, indent=2))  # Resume la salida al usuario.


if __name__ == "__main__":
    main()
