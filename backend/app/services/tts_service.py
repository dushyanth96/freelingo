import asyncio
import base64
import re
import subprocess
import time

import httpx
import openai

from app.core.app_logger import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class KokoroTTSService:
    def __init__(self, base_url: str, voice: str) -> None:
        self.base_url = base_url
        self.voice = voice

    async def health(self) -> None:
        """Raise if Kokoro is unreachable."""
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{self.base_url}/v1/models", timeout=5.0)
            r.raise_for_status()

    async def synthesize(
        self, text: str, voice: str | None = None, language: str | None = None
    ) -> bytes:
        """Call Kokoro-FastAPI and return MP3 audio bytes."""
        _ = language  # Kokoro handles language via the voice model itself
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": text,
                    "voice": voice or self.voice,
                    "response_format": "mp3",
                },
                timeout=30.0,
            )
            response.raise_for_status()
            return response.content


class GeminiTTSService:
    def __init__(
        self,
        *,
        api_key: str,
        model: str | None = None,
        voice: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model or settings.TTS_MODEL
        self.voice = voice or settings.TTS_VOICE

    async def health(self) -> None:
        return None

    @staticmethod
    def _pcm_to_mp3(pcm_audio: bytes, sample_rate: int) -> bytes:
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "s16le",
                    "-ar",
                    str(sample_rate),
                    "-ac",
                    "1",
                    "-i",
                    "pipe:0",
                    "-f",
                    "mp3",
                    "-acodec",
                    "libmp3lame",
                    "pipe:1",
                ],
                input=pcm_audio,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=30,
            )
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError("Gemini TTS audio conversion failed") from exc
        if not result.stdout:
            raise RuntimeError("Gemini TTS audio conversion returned empty audio")
        return result.stdout

    async def _generate_audio(self, text: str, voice: str) -> tuple[bytes, int]:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}
                },
            },
        }
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    params={"key": self.api_key},
                    json=payload,
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            raise RuntimeError("Gemini TTS request failed") from exc
        except (ValueError, TypeError) as exc:
            raise RuntimeError("Gemini TTS returned malformed JSON") from exc

        try:
            inline_data = data["candidates"][0]["content"]["parts"][0]["inlineData"]
            encoded_audio = inline_data["data"]
            mime_type = inline_data.get("mimeType", "audio/L16;rate=24000")
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Gemini TTS returned no audio data") from exc

        try:
            audio = base64.b64decode(encoded_audio, validate=True)
        except (ValueError, TypeError) as exc:
            raise RuntimeError("Gemini TTS returned invalid audio data") from exc
        if not audio:
            raise RuntimeError("Gemini TTS returned empty audio")

        sample_rate_match = re.search(r"rate=(\d+)", mime_type)
        sample_rate = int(sample_rate_match.group(1)) if sample_rate_match else 24000
        return audio, sample_rate

    async def synthesize(
        self, text: str, voice: str | None = None, language: str | None = None
    ) -> bytes:
        _ = language
        text = text.strip()
        if not text:
            logger.warning("[tts-gemini] Empty text received for synthesis")
            return b""
        pcm_audio, sample_rate = await self._generate_audio(text, voice or self.voice)
        return await asyncio.to_thread(self._pcm_to_mp3, pcm_audio, sample_rate)


class OpenAITTSService:
    def __init__(
        self,
        api_key: str,
        model: str,
        voice: str,
        speed: float = 1.0,
        timeout: float | None = None,
    ) -> None:
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self.voice = voice
        self.speed = speed
        self.timeout = timeout

    async def health(self) -> None:
        """Raise if OpenAI TTS is unreachable (lightweight models list call)."""
        await self._client.models.list()

    async def synthesize(
        self, text: str, voice: str | None = None, language: str | None = None
    ) -> bytes:
        """Call OpenAI TTS API and return MP3 audio bytes."""
        _ = language
        text = text.strip()
        if not text:
            logger.warning("[tts-openai] Empty text received for synthesis")
            return b""

        req_voice = (voice or self.voice).strip()
        input_len = len(text)
        start_t = time.perf_counter()
        logger.info(
            "[tts-openai] request_start model=%s voice=%s chars=%d",
            self.model,
            req_voice,
            input_len,
        )
        request_payload = {
            "model": self.model,
            "voice": req_voice,
            "input": text,
            "response_format": "mp3",
            "speed": self.speed,
        }
        if self.timeout is not None:
            request_payload["timeout"] = self.timeout

        response = await self._client.audio.speech.create(**request_payload)
        audio = response.content
        if not audio:
            raise RuntimeError("OpenAI TTS returned empty audio payload")
        elapsed_ms = (time.perf_counter() - start_t) * 1000
        request_id = getattr(response, "request_id", None)
        if request_id is None:
            headers = getattr(response, "headers", None)
            if headers is not None:
                request_id = headers.get("x-request-id")
        logger.info(
            "[tts-openai] request_ok model=%s voice=%s chars=%d bytes=%d ms=%.1f request_id=%s",
            self.model,
            req_voice,
            input_len,
            len(audio),
            round(elapsed_ms, 1),
            request_id,
        )
        return audio
