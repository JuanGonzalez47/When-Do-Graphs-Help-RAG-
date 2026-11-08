# Módulos de `research/src`

Este paquete implementa el pipeline del proyecto; los scripts de `scripts/`
son sus puntos de entrada y las pruebas de `tests/` comprueban las funciones
principales.

## Flujo general

1. `paths.py`, `io.py` y `data.py` localizan archivos, leen datos y cargan la
   muestra de preguntas congelada.
2. `chunking.py` prepara ventanas de texto. `graph_extraction.py`, `llm.py`,
   `prompts.py` y `graph.py` extraen triples, filtran metadatos y guardan
   grafos por fuente.
3. `retrieval.py` implementa búsqueda TF-IDF sobre chunks. `graph_retrieval.py`
   indexa nodos/aristas y selecciona subgrafos. `conditions.py` combina esos
   mecanismos en las condiciones RAG y GraphRAG.
4. `budget.py` limita y cuenta el contexto. `generation.py` forma las
   respuestas; `pipeline.py` ejecuta las celdas y escribe predicciones.
5. `sample_questions.py` es una utilidad de línea de comandos para crear una
   muestra nueva. En las evaluaciones registradas se usa
   `research/configs/questions_400.json`; no debe regenerarse al reproducir
   esos resultados.

## Archivos conservados

| Archivo | Responsabilidad |
|---|---|
| `__init__.py` | Declara el paquete y su versión local. |
| `budget.py` | Conteo de tokens, recorte y empaquetado del contexto. |
| `chunking.py` | División tokenizada y solapada del corpus. |
| `conditions.py` | Construcción de contexto para cada condición experimental. |
| `data.py` | Lectura, validación y muestreo balanceado de preguntas. |
| `generation.py` | Llamada de generación y formato común de predicción. |
| `graph.py` | Normalización, construcción, transformación y métricas de grafos. |
| `graph_extraction.py` | Extracción de triples por chunk y construcción de grafos. |
| `graph_retrieval.py` | Selección full, content, greedy y adaptive de contexto de grafo. |
| `io.py` | Lectura/escritura JSON, JSONL, YAML y hashes. |
| `llm.py` | Cliente compatible con Ollama/OpenAI, reintentos y cache. |
| `paths.py` | Rutas canónicas del proyecto y creación de directorios. |
| `prompts.py` | Plantillas de extracción, respuesta y evaluación. |
| `retrieval.py` | Recuperador léxico TF-IDF de chunks. |
| `sample_questions.py` | Generador CLI de una nueva muestra reproducible. |

`stats.py` se retiró porque no era parte del pipeline ni de los análisis
finales: solo lo usaba su prueba unitaria, mientras que los análisis de Q3/Q4
implementan sus propios procedimientos bootstrap. Esto evita conservar una
segunda implementación que pudiera confundirse con la realmente reportada.
