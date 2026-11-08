"""Ayudantes de lectura/escritura segura y huellas SHA-256."""

from __future__ import annotations

import hashlib  # Produce huellas para cache y comprobaciones de identidad.
import json  # Serializa reportes y checkpoints.
import os  # Reemplaza archivos de forma atómica y limpia temporales.
import tempfile  # Crea temporales junto al destino para permitir replace atómico.
from pathlib import Path  # Normaliza rutas entregadas como texto o Path.
from typing import Any, Iterable  # Tipos flexibles de datos JSON y colecciones.

import yaml  # Lee y escribe configuraciones declaradas en YAML.


def read_json(path: str | Path) -> Any:
    """Lee un archivo UTF-8 y decodifica su contenido JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))  # Propaga errores de archivo o formato al llamador.


def write_json(path: str | Path, data: Any) -> None:
    """Guarda un objeto JSON legible sin dejar un archivo parcial."""
    path = Path(path)  # Permite que la API reciba tanto cadenas como Path.
    path.parent.mkdir(parents=True, exist_ok=True)  # Prepara la jerarquía del destino.
    text = json.dumps(data, indent=2, ensure_ascii=False)  # Mantiene caracteres Unicode legibles.
    atomic_write_text(path, text)  # Sustituye el destino únicamente tras completar la escritura.


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Lee cada línea JSON como un registro independiente."""
    path = Path(path)  # Normaliza la ruta una vez para las operaciones siguientes.
    if not path.exists():
        return []  # La ausencia de checkpoint representa cero registros previos.
    rows = []  # Acumula los objetos decodificados.
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()  # Quita espacios y salto de línea antes de validar.
            if line:
                rows.append(json.loads(line))  # Un JSON inválido genera un error explícito.
    return rows  # Preserva el orden de las líneas originales.


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Añade un objeto como nueva línea JSON sin reemplazar el checkpoint."""
    path = Path(path)  # Normaliza la ruta del archivo destino.
    path.parent.mkdir(parents=True, exist_ok=True)  # Crea carpetas antes de abrirlo en append.
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")  # Una escritura por registro facilita reanudar.


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Escribe una colección completa como JSONL, reemplazando el destino."""
    path = Path(path)  # Normaliza el tipo de la ruta.
    path.parent.mkdir(parents=True, exist_ok=True)  # Asegura que exista el directorio.
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")  # Serializa cada predicción en una línea.


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Lee YAML usando el cargador seguro de PyYAML."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))  # Evita constructores arbitrarios.


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    """Guarda un diccionario YAML respetando el orden de sus claves."""
    path = Path(path)  # Acepta el mismo tipo flexible de ruta que las otras funciones.
    path.parent.mkdir(parents=True, exist_ok=True)  # Crea la carpeta de configuración.
    atomic_write_text(path, yaml.safe_dump(data, sort_keys=False))  # Mantiene legible el orden elegido.


def atomic_write_text(path: str | Path, text: str) -> None:
    """Escribe primero un temporal y lo reemplaza en una sola operación atómica."""
    path = Path(path)  # Convierte el destino a Path para construir el temporal.
    path.parent.mkdir(parents=True, exist_ok=True)  # El temporal debe estar junto al destino.
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")  # Reserva un nombre único.
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:  # Transfiere el descriptor al objeto de archivo.
            handle.write(text)  # Completa el temporal antes de tocar el archivo final.
        os.replace(tmp, path)  # Reemplazo atómico: no expone JSON a medio escribir.
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)  # Limpia el temporal si la escritura o el reemplazo fallaron.


def sha256_text(text: str) -> str:
    """Devuelve el digest SHA-256 hexadecimal del texto UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()  # Huella compacta y reproducible.


def cache_key(*parts: str) -> str:
    """Combina componentes con separador y los convierte en una clave de cache."""
    return sha256_text("\x1f".join(parts))  # El separador evita ambigüedades entre concatenaciones.
