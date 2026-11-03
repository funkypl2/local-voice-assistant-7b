"""
Voice Assistant Orchestrator coordinating VAD -> ASR -> Local 7B LLM -> TTS.
"""
from __future__ import annotations
import time
from typing import List, Dict, Generator, Optional
import numpy as np

from vad.detector import VoiceActivityDetector
from asr.engine import FasterWhisperASR
from llm.client import LocalLLMClient
from tts.engine import TTSEngine
from audio.io import AudioIO


class VoiceAssistant:
    """Full-pipeline local voice assistant."""

    def __init__(
        self,
        llm_client: Optional[LocalLLMClient] = None,
        asr_engine: Optional[FasterWhisperASR] = None,
        tts_engine: Optional[TTSEngine] = None,
        audio_io: Optional[AudioIO] = None,
    ):
        self.llm = llm_client or LocalLLMClient()
        self.asr = asr_engine or FasterWhisperASR()
        self.tts = tts_engine or TTSEngine()
        self.audio = audio_io or AudioIO()
        self.vad = VoiceActivityDetector()
        self.history: List[Dict[str, str]] = []

    def reset_history(self) -> None:
        self.history.clear()

    def step(self, user_audio: Optional[np.ndarray] = None, text_input: Optional[str] = None) -> tuple[str, str]:
        """
        Execute one complete turn of conversation:
        1. ASR on user audio (or use direct text_input)
        2. Stream sentences from local 7B LLM
        3. Synthesize and play each sentence with low latency
        """
        # 1. ASR
        if text_input:
            user_text = text_input.strip()
        elif user_audio is not None and len(user_audio) > 0:
            user_text = self.asr.transcribe(user_audio).strip()
        else:
            return "", ""

        if not user_text:
            return "", ""

        self.history.append({"role": "user", "content": user_text})

        # 2. LLM + Streaming TTS
        full_reply = ""
        for sentence in self.llm.stream_sentences(self.history):
            full_reply += sentence + " "
            audio_data = self.tts.synthesize_to_bytes(sentence)
            if audio_data:
                self.audio.play_audio_bytes(audio_data)

        full_reply = full_reply.strip()
        self.history.append({"role": "assistant", "content": full_reply})
        return user_text, full_reply
