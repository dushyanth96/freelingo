import httpx
import pytest

from app.services.stt_service import RemoteFasterWhisperProvider
from app.services.tts_service import GeminiTTSService


@pytest.mark.asyncio
async def test_remote_faster_whisper_provider_transcribe(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "text": "hello there",
                "language": "en",
                "segments": [{"text": "hello there", "start": 0.0, "end": 1.2}],
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, **kwargs):
            captured["url"] = url
            captured["data"] = kwargs.get("data")
            captured["files"] = kwargs.get("files") is not None
            return FakeResponse()

    monkeypatch.setattr("app.services.stt_service.httpx.AsyncClient", FakeClient)

    provider = RemoteFasterWhisperProvider("https://freelingo-stt.onrender.com/")
    result = await provider.transcribe(b"fake-audio", language="en")

    assert result == "hello there"
    assert captured["url"] == "https://freelingo-stt.onrender.com/transcribe"
    assert captured["data"]["language"] == "en"
    assert captured["files"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectError("connection failed"),
        httpx.ReadTimeout("timed out"),
        httpx.HTTPStatusError(
            "server error",
            request=httpx.Request("POST", "https://stt.example.com/transcribe"),
            response=httpx.Response(503),
        ),
    ],
)
async def test_remote_faster_whisper_provider_normalizes_http_errors(monkeypatch, failure):
    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, **kwargs):
            raise failure

    monkeypatch.setattr("app.services.stt_service.httpx.AsyncClient", FakeClient)

    provider = RemoteFasterWhisperProvider("https://freelingo-stt.onrender.com/")
    with pytest.raises(RuntimeError, match="Remote STT request failed"):
        await provider.transcribe(b"fake-audio", language="en")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (ValueError("invalid json"), "malformed JSON"),
        ({"language": "en", "segments": []}, "invalid text field"),
        ({"text": "   ", "language": "en", "segments": []}, "empty transcription"),
        (["not", "an", "object"], "invalid response"),
    ],
)
async def test_remote_faster_whisper_provider_validates_response(monkeypatch, payload, message):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            if isinstance(payload, Exception):
                raise payload
            return payload

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("app.services.stt_service.httpx.AsyncClient", FakeClient)

    provider = RemoteFasterWhisperProvider("https://freelingo-stt.onrender.com")
    with pytest.raises(RuntimeError, match=message):
        await provider.transcribe(b"fake-audio", language="en")


@pytest.mark.asyncio
async def test_gemini_tts_generates_audio_with_configured_voice_and_model(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "audio/L16;rate=24000",
                                        "data": "cGNtLWF1ZGlv",
                                    }
                                }
                            ]
                        }
                    }
                ]
            }

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, params=None, json=None, timeout=None):
            captured.update({"url": url, "params": params, "json": json, "timeout": timeout})
            return FakeResponse()

    monkeypatch.setattr("app.services.tts_service.httpx.AsyncClient", FakeClient)
    monkeypatch.setattr(
        "app.services.tts_service.GeminiTTSService._pcm_to_mp3",
        lambda self, pcm_audio, sample_rate: b"mp3-audio",
    )

    provider = GeminiTTSService(
        api_key="test-key",
        model="gemini-2.5-flash-preview-tts",
        voice="Kore",
    )
    audio = await provider.synthesize("Hello there")

    assert audio == b"mp3-audio"
    assert captured["url"].endswith("/models/gemini-2.5-flash-preview-tts:generateContent")
    assert captured["params"] == {"key": "test-key"}
    assert captured["json"]["contents"][0]["parts"][0]["text"] == "Hello there"
    assert captured["json"]["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert (
        captured["json"]["generationConfig"]["speechConfig"]["voiceConfig"]
        ["prebuiltVoiceConfig"]["voiceName"]
        == "Kore"
    )


@pytest.mark.asyncio
async def test_gemini_tts_returns_empty_for_empty_text():
    provider = GeminiTTSService(api_key="test-key")
    assert await provider.synthesize("  ") == b""


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [httpx.ConnectError("network"), httpx.ReadTimeout("timeout")])
async def test_gemini_tts_wraps_network_errors(monkeypatch, error):
    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, **kwargs):
            raise error

    monkeypatch.setattr("app.services.tts_service.httpx.AsyncClient", FakeClient)

    provider = GeminiTTSService(api_key="test-key")
    with pytest.raises(RuntimeError, match="Gemini TTS request failed"):
        await provider.synthesize("Hello there")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"candidates": []},
        {"candidates": [{"content": {"parts": []}}]},
        {
            "candidates": [
                {"content": {"parts": [{"inlineData": {"data": "not-base64"}}]}}
            ]
        },
    ],
)
async def test_gemini_tts_rejects_unexpected_audio_response(monkeypatch, payload):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("app.services.tts_service.httpx.AsyncClient", FakeClient)

    provider = GeminiTTSService(api_key="test-key")
    with pytest.raises(RuntimeError):
        await provider.synthesize("Hello there")
