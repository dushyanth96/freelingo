import io
import time

import httpx
import openai

from app.core.app_logger import get_logger

logger = get_logger(__name__)


class WhisperSTTService:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def health(self) -> None:
        """Raise if Whisper ASR is unreachable."""
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{self.base_url}/", timeout=5.0)
            r.raise_for_status()

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        mime_type: str = "audio/wav",
        *,
        language: str,
    ) -> str:
        """Send audio to Whisper ASR and return the transcribed text.

        Compatible with onerahmet/openai-whisper-asr-webservice which exposes
        POST /asr?output=json&language=<code> (not the OpenAI /v1/audio/transcriptions path).
        """
        async with httpx.AsyncClient() as client:
            logger.debug(
                "[stt] POST /asr — %d bytes, filename=%s lang=%s",
                len(audio_bytes),
                filename,
                language,
            )
            response = await client.post(
                f"{self.base_url}/asr",
                params={"output": "json", "language": language, "task": "transcribe"},
                files={"audio_file": (filename, audio_bytes, mime_type)},
                timeout=60.0,
            )
            logger.debug("[stt] Response status: %s", response.status_code)
            response.raise_for_status()
            data = response.json()
            text = data.get("text", "").strip()
            logger.info("[stt] Transcribed: %r", text)
            return text


class RemoteFasterWhisperProvider:
    def __init__(self, base_url: str, *, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def health(self) -> None:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/health", timeout=5.0)
            response.raise_for_status()

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        mime_type: str = "audio/wav",
        *,
        language: str,
    ) -> str:
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/transcribe",
                    data={"language": language},
                    files={"file": (filename, audio_bytes, mime_type)},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as exc:
            response = getattr(exc, "response", None)
            logger.warning(
                "[stt-remote] request failed status=%s",
                getattr(response, "status_code", None),
            )
            raise RuntimeError("Remote STT request failed") from exc
        except (ValueError, TypeError) as exc:
            raise RuntimeError("Remote STT returned malformed JSON") from exc

        if not isinstance(payload, dict):
            raise RuntimeError("Remote STT returned an invalid response")

        text = payload.get("text")
        if not isinstance(text, str):
            raise RuntimeError("Remote STT returned an invalid text field")
        text = text.strip()
        if not text:
            raise RuntimeError("Remote STT returned an empty transcription")

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "[stt-remote] duration_ms=%.1f bytes=%d text_len=%d language=%s",
            duration_ms,
            len(audio_bytes),
            len(text),
            language,
        )
        return text


class OpenAISTTService:
    def __init__(self, api_key: str, model: str) -> None:
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model

    async def health(self) -> None:
        """Raise if OpenAI STT is unreachable (lightweight models list call)."""
        await self._client.models.list()

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        mime_type: str = "audio/wav",
        *,
        language: str,
    ) -> str:
        """Transcribe audio using OpenAI Whisper API."""
        audio_file = (filename, io.BytesIO(audio_bytes), mime_type)
        response = await self._client.audio.transcriptions.create(
            model=self.model,
            file=audio_file,
            language=language,
            timeout=60.0,
        )
        text = response.text.strip()
        logger.info(
            "[stt-openai] Transcribed model=%s lang=%s: %r",
            self.model,
            language,
            text,
        )
        return text
