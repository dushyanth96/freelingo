import asyncio
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("stt-service")

MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny")
DEVICE = os.getenv("DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(50 * 1024 * 1024)))

app = FastAPI(title="FreeLingo STT Service")
_model: WhisperModel | None = None


def load_model() -> WhisperModel:
    global _model
    if _model is not None:
        return _model

    logger.info("[stt] loading model model=%s device=%s compute_type=%s", MODEL_NAME, DEVICE, COMPUTE_TYPE)
    _model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
    logger.info("[stt] model_loaded model=%s", MODEL_NAME)
    return _model


def _transcribe_file(model: WhisperModel, audio_path: str, language: str) -> dict[str, Any]:
    segments, info = model.transcribe(audio_path, language=language, beam_size=5)
    segments_list = []
    text_parts = []
    for segment in segments:
        segment_data = {
            "id": segment.id,
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text.strip(),
        }
        segments_list.append(segment_data)
        text_parts.append(segment.text.strip())

    return {
        "text": " ".join(part for part in text_parts if part).strip(),
        "language": getattr(info, "language", None) or language,
        "segments": segments_list,
    }


@app.on_event("startup")
def startup() -> None:
    load_model()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE, "compute_type": COMPUTE_TYPE}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("en"),
) -> JSONResponse:
    if file is None or file.filename is None:
        raise HTTPException(status_code=400, detail="Audio file is required")

    start_time = time.perf_counter()
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Audio file is empty")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file too large")

    suffix = Path(file.filename).suffix or ".audio"
    temp_path: str | None = None

    try:
        model = load_model()
        transcription_start = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as audio_file:
            audio_file.write(data)
            temp_path = audio_file.name

        payload = await asyncio.to_thread(_transcribe_file, model, temp_path, language)
        duration_ms = (time.perf_counter() - transcription_start) * 1000
        total_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "[stt] transcription_ok duration_ms=%.1f total_ms=%.1f text_length=%d segments=%d",
            duration_ms,
            total_ms,
            len(payload["text"]),
            len(payload["segments"]),
        )
        return JSONResponse(payload)
    except Exception as exc:
        logger.exception("[stt] transcription_error")
        raise HTTPException(status_code=500, detail=f"STT transcription failed: {exc}") from exc
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
