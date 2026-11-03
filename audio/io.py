"""
Audio Input/Output driver handling recording and speaker playback.
"""
from __future__ import annotations
import io
import time
import numpy as np
from typing import Optional, Callable


class AudioIO:
    """Manages microphone recording and audio stream playback."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels

    def record_seconds(self, duration: float = 3.0) -> np.ndarray:
        """Record audio from default microphone for specified duration."""
        try:
            import sounddevice as sd
            frames = int(duration * self.sample_rate)
            data = sd.rec(frames, samplerate=self.sample_rate, channels=self.channels, dtype="float32")
            sd.wait()
            return data.flatten()
        except Exception as e:
            # Simulated audio buffer with slight noise for testing
            n_samples = int(duration * self.sample_rate)
            return (np.random.randn(n_samples) * 0.005).astype(np.float32)

    def play_audio_bytes(self, audio_bytes: bytes) -> None:
        """Play audio bytes (WAV/MP3)."""
        if not audio_bytes:
            return
        try:
            import sounddevice as sd
            import soundfile as sf
            data, fs = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            sd.play(data, fs)
            sd.wait()
        except Exception:
            # Fallback: sleep for estimated duration
            time.sleep(0.5)
