import io
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import main


class FakeModel:
    def __init__(self) -> None:
        self.received_path: str | None = None

    def transcribe(self, audio_path: str, *, language: str, beam_size: int):
        self.received_path = audio_path
        with open(audio_path, "rb") as audio_file:
            assert audio_file.read() == b"encoded-audio"
        return (
            [SimpleNamespace(id=0, start=0.0, end=1.0, text=" hello there ")],
            SimpleNamespace(language=language),
        )


def test_health_and_transcribe_contract(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(main, "load_model", lambda: model)

    with TestClient(main.app) as client:
        health = client.get("/health")
        response = client.post(
            "/transcribe",
            files={"file": ("recording.webm", io.BytesIO(b"encoded-audio"), "audio/webm")},
            data={"language": "en"},
        )

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert response.status_code == 200
    assert response.json() == {
        "text": "hello there",
        "language": "en",
        "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "hello there"}],
    }
    assert model.received_path is not None
    assert not main.Path(model.received_path).exists()


@pytest.mark.parametrize(
    ("payload", "expected_status"),
    [(b"", 400), (b"oversized", 413)],
)
def test_transcribe_rejects_empty_and_oversized_uploads(monkeypatch, payload, expected_status):
    monkeypatch.setattr(main, "load_model", lambda: FakeModel())
    monkeypatch.setattr(main, "MAX_AUDIO_BYTES", 1 if payload else 50 * 1024 * 1024)

    with TestClient(main.app) as client:
        response = client.post(
            "/transcribe",
            files={"file": ("recording.wav", io.BytesIO(payload), "audio/wav")},
            data={"language": "en"},
        )

    assert response.status_code == expected_status


def test_transcribe_cleans_up_temporary_file_on_error(monkeypatch):
    class ErrorModel:
        received_path: str | None = None

        def transcribe(self, audio_path: str, *, language: str, beam_size: int):
            self.received_path = audio_path
            raise RuntimeError("decode failed")

    model = ErrorModel()
    monkeypatch.setattr(main, "load_model", lambda: model)

    with TestClient(main.app, raise_server_exceptions=False) as client:
        response = client.post(
            "/transcribe",
            files={"file": ("recording.mp3", io.BytesIO(b"encoded-audio"), "audio/mpeg")},
            data={"language": "en"},
        )

    assert response.status_code == 500
    assert model.received_path is not None
    assert not main.Path(model.received_path).exists()