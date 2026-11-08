#!/usr/bin/env python
"""Quita aristas de metadatos de los grafos y puede crear una copia previa.

ATENCIÓN: modifica ``research/graphs`` en el sitio. Si se usa para reproducir
el procesamiento, hay que conservar los grafos originales y el informe que
registra cuántas aristas y nodos fueron eliminados.
"""

# Permite ejecutar este archivo directamente desde la raíz del proyecto.
from __future__ import annotations

# argparse procesa opciones de terminal; shutil copia directorios.
import argparse
import shutil
import sys
from pathlib import Path

# ROOT identifica la carpeta del proyecto independientemente del directorio actual.
ROOT = Path(__file__).resolve().parents[1]
# Añade el paquete de investigación al buscador de módulos de Python.
sys.path.insert(0, str(ROOT / "research"))

# Reutiliza la regla compartida para detectar triples de metadatos.
from src.graph import is_metadata_triple  
# Reutiliza las funciones centrales de lectura y escritura JSON.
from src.io import read_json, write_json  
# Rutas canónicas de grafos procesados y reportes del proyecto.
from src.paths import GRAPHS, RESULTS  


def clean_graph(data: dict, source: str) -> tuple[dict, int, int]:
    """Devuelve el grafo limpio, cantidad de aristas y cantidad de nodos quitados."""
    # Se construye una lista nueva para no conservar accidentalmente aristas filtradas.
    edges = []
    # Este contador permitirá documentar el impacto de la limpieza.
    removed = 0
    # Revisa cada arista guardada en el JSON del grafo.
    for edge in data.get("edges", []):
        # Algunos formatos guardan varias relaciones para la misma pareja de nodos.
        relations = edge.get("relations") or [""]
        # Si alguna relación describe metadatos del libro/corpus, se descarta la arista.
        if any(is_metadata_triple(edge["source"], relation, edge["target"], source) for relation in relations):
            removed += 1
            continue
        # Las aristas que pasan el filtro se mantienen sin modificar sus atributos.
        edges.append(edge)
    # Solo se conservan nodos que todavía aparecen como extremo de una arista.
    used = {edge["source"] for edge in edges} | {edge["target"] for edge in edges}
    # El campo id identifica cada nodo en el formato JSON del proyecto.
    nodes = [node for node in data.get("nodes", []) if node["id"] in used]
    # El resultado también informa aristas quitadas y nodos que quedaron aislados.
    return {"nodes": nodes, "edges": edges}, removed, len(data.get("nodes", [])) - len(nodes)


def main() -> None:
    """Procesa todos los grafos indicados y escribe un informe de limpieza."""
    # Define la interfaz para poder seleccionar rutas y omitir explícitamente el respaldo.
    parser = argparse.ArgumentParser(description="Remove extraction artifacts from existing graphs")
    # Directorio de entrada; por defecto es la carpeta principal de grafos.
    parser.add_argument("--graphs-dir", type=Path, default=GRAPHS)
    # Carpeta donde se copiarán los originales antes de modificarlos.
    parser.add_argument("--backup-dir", type=Path, default=ROOT / "research" / "graphs_raw_backup")
    # Esta opción desactiva la copia; es peligrosa si no hay otra copia segura.
    parser.add_argument("--no-backup", action="store_true")
    # Convierte los argumentos de terminal en valores accesibles para el programa.
    args = parser.parse_args()

    # La copia solo se crea si no existe; no actualiza un respaldo antiguo.
    if not args.no_backup and not args.backup_dir.exists():
        # copytree falla si el destino ya existe, evitando sobrescribirlo silenciosamente.
        shutil.copytree(args.graphs_dir, args.backup_dir)
        # Deja constancia de la ruta usada para el respaldo.
        print(f"backup -> {args.backup_dir}")

    # Acumula cifras por grafo para guardar un reporte auditable.
    report = {}
    # Totales globales de elementos eliminados en todos los grafos.
    total_removed = 0
    total_nodes = 0
    # Los grafos se encuentran en subcarpetas por split, por ejemplo novel/ y medical/.
    for path in sorted(args.graphs_dir.glob("*/*.json")):
        # Carga el archivo JSON de un grafo.
        data = read_json(path)
        # Aplica el filtro y recupera cuántos elementos se eliminaron.
        cleaned, removed, dropped_nodes = clean_graph(data, path.stem)
        # Reemplaza el grafo original en el mismo sitio: esta es la operación destructiva.
        write_json(path, cleaned)
        # Guarda los tamaños finales y las cantidades eliminadas para este grafo.
        report[path.stem] = {
            "edges": len(cleaned["edges"]),
            "removed_edges": removed,
            "nodes": len(cleaned["nodes"]),
            "dropped_nodes": dropped_nodes,
        }
        # Actualiza los totales del conjunto completo.
        total_removed += removed
        total_nodes += dropped_nodes
        # Muestra el progreso para identificar qué fuentes se modificaron.
        print(f"{path.parent.name}/{path.name}: -{removed} edges, -{dropped_nodes} nodes", flush=True)

    # Escribe el resultado detallado en research/results/graph_cleanup.json.
    write_json(RESULTS / "graph_cleanup.json", {"graphs": report, "removed_edges": total_removed, "dropped_nodes": total_nodes})
    # Resume el impacto total de la limpieza en la terminal.
    print(f"removed {total_removed} edges and {total_nodes} nodes")


# Solo se ejecuta main al invocar este archivo, no al importarlo como módulo.
if __name__ == "__main__":
    main()
