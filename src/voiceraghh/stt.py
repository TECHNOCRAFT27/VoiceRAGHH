from dataclasses import dataclass
from dotenv import load_dotenv
import os
import requests


load_dotenv()


@dataclass
class Transcription:
    text: str
    language: str
    confidence: float


def transcribe_file(audio_path: str, language: str | None = None) -> Transcription:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY not set")
    
    with open(audio_path, "rb") as f:
        response = requests.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": api_key},
            files={"file": (audio_path, f, "audio/wav")},
            data={"model_id": "scribe_v1"}
        )
    
    response.raise_for_status()
    result = response.json()
    
    return Transcription(
        text=result.get("text", ""),
        language=result.get("language", language or "en"),
        confidence=1.0
    )


def transcribe_bytes(audio_bytes: bytes, filename: str = "audio.wav", language: str | None = None) -> Transcription:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY not set")
    
    response = requests.post(
        "https://api.elevenlabs.io/v1/speech-to-text",
        headers={"xi-api-key": api_key},
        files={"file": (filename, audio_bytes, "audio/wav")},
        data={"model_id": "scribe_v1"}
    )
    
    response.raise_for_status()
    result = response.json()
    
    return Transcription(
        text=result.get("text", ""),
        language=result.get("language", language or "en"),
        confidence=1.0
    )
