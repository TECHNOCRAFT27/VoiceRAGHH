import numpy as np
import wave
import tempfile
import os
from voiceraghh.stt import transcribe_file


def generate_test_audio(duration: float = 1.0, sample_rate: int = 16000) -> str:
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * 440 * t) * 0.5
    audio = (audio * 32767).astype(np.int16)
    
    path = tempfile.mktemp(suffix=".wav")
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return path


if __name__ == "__main__":
    print("Generating test audio (440Hz tone)...")
    audio_path = generate_test_audio(duration=2.0)
    
    print("Transcribing...")
    result = transcribe_file(audio_path, size="tiny")
    
    print(f"Text: {result.text}")
    print(f"Language: {result.language}")
    print(f"Confidence: {result.confidence:.2f}")
    
    os.unlink(audio_path)
