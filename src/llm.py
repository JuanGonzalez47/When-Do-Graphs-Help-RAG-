"""Cliente de chat compatible con OpenAI, cache y lectura robusta de JSON."""

from __future__ import annotations

import json  # Decodifica las respuestas estructuradas.
import os  # Lee credenciales y opciones desde variables de entorno.
import re  # Quita cercas Markdown alrededor de JSON.
import time  # Espera entre intentos fallidos del servicio.
from pathlib import Path  # Forma rutas para la cache persistente.
from typing import Any  # El cliente externo y las respuestas pueden tener distintas formas.

from .io import cache_key, read_json, write_json  # Cachea solicitudes con una huella estable.
from .paths import CACHE  # Ubicación común para datos regenerables.

DEFAULT_CACHE_DIR = CACHE / "llm"  # Respuestas guardadas por prompt/modelo.
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"  # Endpoint local compatible con OpenAI.
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"  # Modelo local por omisión en este proyecto.
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"  # Modelo remoto si se selecciona OpenAI.
OPENAI_MODEL_ALIASES = {"", "gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"}  # Valores tratados como selección genérica al usar Ollama.


def looks_local(url: str | None) -> bool:
    """Detecta señales conocidas de una URL local de Ollama."""
    if not url:
        return False  # Una URL ausente no demuestra que el servicio sea local.
    lowered = url.lower()  # Ignora diferencias de mayúsculas en la dirección.
    return any(marker in lowered for marker in ("localhost", "127.0.0.1", "0.0.0.0", "11434"))  # Busca host/puerto habituales.


def extract_json(text: str) -> Any:
    """Interpreta JSON completo o extrae el primer objeto/arreglo válido de la respuesta."""
    cleaned = text.strip()  # Elimina espacios que no pertenecen al documento.
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)  # Quita cerca inicial Markdown.
    cleaned = re.sub(r"\s*```$", "", cleaned)  # Quita cerca final si el modelo la agregó.
    try:
        return json.loads(cleaned)  # Camino normal: la respuesta entera ya es JSON válido.
    except json.JSONDecodeError:
        pass  # Si hay texto adicional, intenta rescatar un valor JSON incrustado.
    decoder = json.JSONDecoder()  # raw_decode permite ignorar prefijos explicativos del modelo.
    for index, character in enumerate(cleaned):
        if character not in "[{":
            continue  # Solo un arreglo u objeto puede ser el comienzo del valor buscado.
        try:
            value, _ = decoder.raw_decode(cleaned[index:])  # Prueba decodificar desde esta llave/corchete.
            return value  # Devuelve el primer objeto/arreglo completo que se pueda leer.
        except json.JSONDecodeError:
            continue  # Puede tratarse de una llave dentro de texto no estructurado.
    raise ValueError(f"Could not parse JSON from response: {text[:200]!r}")  # Falla con una muestra útil para depuración.


class LLMClient:
    """Envía prompts a una API compatible, reintenta fallos y reutiliza cache."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        base_url: str | None = None,
        api_key: str | None = None,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        client: Any | None = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        provider: str | None = None,
    ) -> None:
        # Los argumentos explícitos tienen prioridad sobre la configuración del entorno.
        api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")  # Busca credencial genérica y luego OpenAI.
        base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OLLAMA_BASE_URL")  # Permite escoger endpoint desde el shell.
        provider = provider or os.getenv("LLM_PROVIDER")  # Lee proveedor si no se pasó al constructor.
        if provider is None:
            if not api_key or looks_local(base_url):
                provider = "ollama"  # Sin credencial o con URL local, usa Ollama.
            else:
                provider = "openai"  # En otro caso, asume el proveedor remoto.
        provider = provider.lower()  # Hace que la comparación no dependa de mayúsculas.

        if provider == "ollama":
            base_url = base_url or DEFAULT_OLLAMA_BASE_URL  # Aplica el endpoint local por omisión.
            api_key = api_key or "ollama"  # Ollama suele aceptar cualquier cadena no vacía como clave.
            if model is None or model in OPENAI_MODEL_ALIASES:
                model = os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL  # Sustituye alias remotos por el modelo local.
        else:
            if model is None or model == "":
                model = DEFAULT_OPENAI_MODEL  # Selecciona modelo remoto solo si no se indicó otro.

        self.provider = provider  # Proveedor ya inferido o configurado.
        self.model = str(model)  # Guarda el nombre como cadena para solicitudes y cache.
        self.temperature = temperature  # Control de aleatoriedad por omisión.
        self.max_tokens = max_tokens  # Tope por omisión de tokens generados.
        self.base_url = base_url  # Dirección del endpoint.
        self.api_key = api_key  # Credencial; no se escribe dentro de la cache.
        self.cache_dir = Path(cache_dir)  # Directorio para respuestas repetidas.
        self.client = client  # Permite inyectar un cliente falso en pruebas.
        self.max_retries = max_retries  # Cantidad máxima de intentos ante fallos.
        self.retry_delay = retry_delay  # Segundos base entre reintentos.

    def _client(self) -> Any:
        if self.client is None:
            from openai import OpenAI  # Import diferido: no se requiere si el test inyecta un cliente.

            if not self.api_key:
                raise RuntimeError(
                    "No API key found and no local endpoint configured. "
                    "Set LLM_API_KEY/OPENAI_API_KEY, or run Ollama and use "
                    "LLM_PROVIDER=ollama with OLLAMA_BASE_URL."
                )
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)  # Crea el cliente para el endpoint elegido.
        return self.client  # Reutiliza una instancia para las llamadas siguientes.

    def _call(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        """Realiza una sola llamada, sin reintentos ni lectura/escritura de cache."""
        client = self._client()  # Obtiene o inicializa el cliente compartido.
        response = client.chat.completions.create(
            model=self.model,  # Modelo configurado en el constructor.
            messages=[
                {"role": "system", "content": system},  # Instrucciones permanentes.
                {"role": "user", "content": user},  # Pregunta, texto o contexto variable.
            ],
            temperature=temperature,  # Parámetro de muestreo de la generación.
            max_tokens=max_tokens,  # Límite de longitud de la respuesta.
        )
        return str(response.choices[0].message.content or "")  # Normaliza respuesta vacía a cadena.

    def complete(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        use_cache: bool = True,
    ) -> str:
        """Obtiene una respuesta desde cache o ejecuta llamadas con reintentos."""
        temperature = self.temperature if temperature is None else temperature  # Usa el parámetro por llamada o el predeterminado.
        max_tokens = self.max_tokens if max_tokens is None else max_tokens  # Resuelve el límite efectivo de salida.
        key = cache_key(
            self.model,  # Cambiar modelo genera una entrada de cache diferente.
            str(temperature),  # La temperatura afecta la respuesta y forma parte de la identidad.
            str(max_tokens),  # El límite de salida también afecta el resultado.
            system,  # Incluye todas las instrucciones del sistema en la huella.
            user,  # Incluye el prompt del usuario para evitar colisiones entre preguntas.
        )
        cache_path = self.cache_dir / f"{key}.json"  # Un archivo independiente por solicitud.
        if use_cache and cache_path.exists():
            cached = read_json(cache_path)  # Reutiliza respuesta previa si el archivo está íntegro.
            if isinstance(cached, dict) and "text" in cached:
                return str(cached["text"])  # No vuelve a consumir una llamada al modelo.

        last_error: Exception | None = None  # Guarda la última excepción para adjuntarla al error final.
        for attempt in range(self.max_retries):
            try:
                text = self._call(system, user, temperature, max_tokens)  # Intenta llamar al endpoint una vez.
                if use_cache:
                    write_json(
                        cache_path,
                        {
                            "model": self.model,  # Permite auditar el modelo de la respuesta cacheada.
                            "temperature": temperature,  # Registra el parámetro de generación.
                            "max_tokens": max_tokens,  # Registra el límite utilizado.
                            "system": system,  # Conserva prompt de sistema para reproducibilidad.
                            "user": user,  # Conserva prompt de usuario para inspección.
                            "text": text,  # Contenido que se devolverá en futuros aciertos de cache.
                        },
                    )
                return text  # La llamada fue exitosa; no ejecuta los demás reintentos.
            except Exception as error:
                last_error = error  # Conserva la causa para el mensaje final.
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_delay * (attempt + 1))  # Aumenta la espera en cada nuevo intento.
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts") from last_error  # Nunca simula éxito si fallan todas.

    def complete_json(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        use_cache: bool = True,
    ) -> Any:
        """Genera texto con complete() y lo interpreta como JSON."""
        text = self.complete(
            system,  # Instrucción para el proveedor.
            user,  # Prompt que recibirá el modelo.
            temperature=temperature,  # Reenvía la opción si se especificó.
            max_tokens=max_tokens,  # Reenvía el límite si se especificó.
            use_cache=use_cache,  # Mantiene la misma política de cache.
        )
        return extract_json(text)  # Falla explícitamente si la respuesta no contiene JSON válido.
