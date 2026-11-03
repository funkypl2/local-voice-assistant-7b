"""
Unit tests for Local-VoiceAssistant-7B components.
"""
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pytest
import numpy as np
from vad.detector import VoiceActivityDetector
from asr.engine import FasterWhisperASR
from llm.client import LocalLLMClient
from tts.engine import TTSEngine
from core.assistant import VoiceAssistant


def test_vad_silence_and_speech():
    vad = VoiceActivityDetector(sample_rate=16000, energy_threshold=0.01)
    silence = np.zeros(16000, dtype=np.float32)
    is_speaking, finished = vad.process_chunk(silence)
    assert not is_speaking

    speech = (np.sin(2 * np.pi * 300 * np.linspace(0, 1, 16000)) * 0.5).astype(np.float32)
    is_speaking, finished = vad.process_chunk(speech)
    assert is_speaking


def test_llm_sentence_streaming():
    client = LocalLLMClient()
    # Test streaming sentences
    sentences = list(client.stream_sentences([{"role": "user", "content": "hello"}]))
    assert len(sentences) >= 1
    assert any("7B" in s or "语音助手" in s for s in sentences)


def test_tts_engine():
    tts = TTSEngine()
    audio_bytes = tts.synthesize_to_bytes("你好，世界！")
    assert len(audio_bytes) > 0


def test_full_assistant_step():
    assistant = VoiceAssistant()
    u, a = assistant.step(text_input="测试问答")
    assert u == "测试问答"
    assert len(a) > 0
    assert len(assistant.history) == 2
