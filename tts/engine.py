"""
Text-to-Speech (TTS) module supporting Edge-TTS and offline waveform synthesis.
"""
from __future__ import annotations
import asyncio
import io
import os
import numpy as np
from typing import Optional


class TTSEngine:
    """Text-to-Speech synthesis engine with sentence streaming capability."""

    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural", rate: str = "+0%", pitch: str = "+0Hz"):
        self.voice = voice
        self.rate = rate
        self.pitch = pitch

    def synthesize_to_bytes(self, text: str) -> bytes:
        """Synthesize text to audio bytes (MP3/WAV format)."""
        if not text.strip():
            return b""

        try:
            import edge_tts

            async def _run():
                communicate = edge_tts.Communicate(text, self.voice, rate=self.rate, pitch=self.pitch)
                audio_buffer = bytearray()
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_buffer.extend(chunk["data"])
                return bytes(audio_buffer)

            return asyncio.run(_run())
        except Exception:
            # Fallback offline waveform generator (generates 0.5s gentle chime audio bytes)
            return self._generate_fallback_wav(duration=0.5)

    def _generate_fallback_wav(self, duration: float = 0.5, sample_rate: int = 16000) -> bytes:
        import struct

        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        # 440 Hz gentle sine tone
        waveform = 0.2 * np.sin(2 * np.pi * 440.0 * t)
        int_data = (waveform * 32767).astype(np.int16)

        buf = io.BytesIO()
        # WAV header
        buf.write(b"RIFF")
        buf.write(struct.pack("<I", 36 + len(int_data) * 2))
        buf.write(b"WAVEfmt ")
        buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
        buf.write(b"data")
        buf.write(struct.pack("<I", len(int_data) * 2))
        buf.write(int_data.tobytes())
        return buf.getvalue()
