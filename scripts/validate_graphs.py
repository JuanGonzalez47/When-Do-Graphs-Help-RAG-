#!/usr/bin/env python
"""Comprueba que los grafos esperados existan, se puedan leer y no estén vacíos.

Las estadísticas son controles estructurales; no demuestran que los grafos sean
semánticamente correctos ni que las relaciones extraídas sean verdaderas.
"""

# Importaciones futuras de anotaciones: permiten usar tipos modernos con compatibilidad.
from __future__ import annotations

# json produce el informe legible que se guarda al final.
import json
# sys permite añadir el paquete de investigación al path de Python.
import sys
# Path construye rutas de archivos sin concatenar texto manualmente.
from pathlib import Path

# Calcula la raíz del proyecto a partir de la ubicación de este script.
ROOT = Path(__file__).resolve().parents[1]
# Hace que las importaciones src.* funcionen al ejecutar scripts/validate_graphs.py.
sys.path.insert(0, str(ROOT / "research"))

# Carga el inventario de fuentes que deben tener grafo.
from src.data import load_corpus  
# Funciones que leen un grafo y calculan medidas estructurales.
from src.graph import graph_metrics, load_graph  
# Convierte el nombre de una fuente al nombre seguro usado para el archivo.
from src.graph_extraction import safe_name  
# Directorios estándar de grafos y reportes.
from src.paths import GRAPHS, RESULTS  


def main() -> None:
    """Revisa cada fuente y escribe un resumen con errores y métricas."""
    # El informe agrupa filas por split, incidencias y totales acumulados.
    report = {"splits": {}, "errors": [], "totals": {}}
    # Contadores generales para Novel y Medical combinados.
    grand_nodes = 0
    grand_edges = 0
    grand_sources = 0
    # Los datos del benchmark se organizaron en estos dos conjuntos.
    for split in ("novel", "medical"):
        # Cada clave del corpus es una fuente para la que se espera un archivo de grafo.
        corpus = load_corpus(split)
        # Filas con métricas correspondientes solo a este split.
        rows = []
        # Ordenar fuentes vuelve estable y fácil de comparar el reporte.
        for source in sorted(corpus):
            # safe_name garantiza que el nombre de la fuente coincida con el archivo creado.
            path = GRAPHS / split / f"{safe_name(source)}.json"
            # La ausencia de una fuente se reporta y no detiene la auditoría completa.
            if not path.exists():
                report["errors"].append(f"missing graph: {path}")
                continue
            try:
                # Intenta leer el grafo y calcular sus medidas estructurales.
                graph = load_graph(path)
                metrics = graph_metrics(graph)
            # Un archivo ilegible se registra; luego se continúa con la siguiente fuente.
            except Exception as error:
                report["errors"].append(f"failed to load {path}: {error!r}")
                continue
            # Guarda métricas descriptivas para inspección, no una calificación de calidad.
            rows.append(
                {
                    "source": source,
                    "path": str(path),
                    "nodes": metrics["node_count"],
                    "edges": metrics["edge_count"],
                    "average_degree": round(metrics["average_degree"], 4),
                    "average_clustering": round(metrics["average_clustering"], 4),
                    "components": metrics["components"],
                    "largest_component": metrics["largest_component"],
                    "isolated_nodes": metrics["isolated_nodes"],
                }
            )
            # Suma las métricas de esta fuente a los totales globales.
            grand_nodes += metrics["node_count"]
            grand_edges += metrics["edge_count"]
            grand_sources += 1
            # Un grafo sin nodos o aristas no puede alimentar las condiciones GraphRAG.
            if metrics["node_count"] == 0 or metrics["edge_count"] == 0:
                report["errors"].append(f"empty graph: {source}")
        # Resume cantidad de fuentes esperada/encontrada y sus nodos/aristas por split.
        report["splits"][split] = {
            "sources_expected": len(corpus),
            "sources_found": len(rows),
            "nodes": sum(row["nodes"] for row in rows),
            "edges": sum(row["edges"] for row in rows),
            "rows": rows,
        }
    # Añade el total de fuentes y elementos acumulados en ambos splits.
    report["totals"] = {
        "sources": grand_sources,
        "nodes": grand_nodes,
        "edges": grand_edges,
    }
    # Asegura que la carpeta de reportes exista antes de escribir.
    RESULTS.mkdir(parents=True, exist_ok=True)
    # Ubicación estándar del reporte de validación.
    out = RESULTS / "graph_validation.json"
    # Persiste todos los datos para consultarlos posteriormente.
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Imprime un resumen compacto, incluyendo los errores y la ruta del reporte.
    print(json.dumps({"totals": report["totals"], "errors": report["errors"], "report": str(out)}, indent=2))
    # Código de salida distinto de cero hace fallar automatizaciones si hay problemas.
    if report["errors"]:
        raise SystemExit(1)


# Impide ejecutar la auditoría cuando este módulo solo se importa.
if __name__ == "__main__":
    main()
