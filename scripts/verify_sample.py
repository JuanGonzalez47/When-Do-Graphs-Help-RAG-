#!/usr/bin/env python
"""Verifica balance e identidad de la muestra experimental congelada de 400 preguntas."""

# Activa anotaciones de tipos compatibles con versiones anteriores de Python.
from __future__ import annotations

# hashlib calcula el digest SHA-256 de los IDs.
import hashlib
# json imprime el reporte de validación con formato legible.
import json
# sys y Path permiten importar el código compartido de investigación.
import sys
from pathlib import Path

# Obtiene la carpeta raíz del proyecto desde la ubicación de este archivo.
ROOT = Path(__file__).resolve().parents[1]
# Añade research/ para poder importar el paquete src.
sys.path.insert(0, str(ROOT / "research"))

# Carga la muestra congelada y sus reglas de validación.
from src.data import load_sample, validate_sample 


def main() -> None:
    """Comprueba los IDs, la estratificación y el hash guardado de la muestra."""
    # Carga registros, IDs ordenados y digest que identifican la muestra oficial del proyecto.
    sample = load_sample()
    # Convierte los registros a lista para validarlos y recorrerlos de forma explícita.
    records = list(sample["records"])
    # Comprueba que tamaños, tipos y splits satisfagan las condiciones esperadas.
    report = validate_sample(records)
    # Une IDs en su orden congelado y calcula su huella criptográfica.
    digest = hashlib.sha256("\n".join(sample["question_ids"]).encode()).hexdigest()
    # Incluso cambiar el orden altera el hash y señala que la muestra ya no coincide.
    if digest != sample["sha256_ids"]:
        # El error no debe parecer una validación exitosa.
        raise SystemExit("Sample hash mismatch")
    # Imprime digest y resumen; no modifica ni regenera la muestra.
    print(json.dumps({"sha256_ids": digest, "report": report}, indent=2))


# Ejecuta main solo cuando se llama al script desde terminal.
if __name__ == "__main__":
    main()
