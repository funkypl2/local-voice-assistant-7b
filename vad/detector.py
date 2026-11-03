"""
Voice Activity Detection (VAD) module using energy thresholding with optional Silero VAD backend.
"""
import numpy as np
from typing import Optional


class VoiceActivityDetector:
    """Detects start and end of speech in audio chunks."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        energy_threshold: float = 0.015,
        silence_duration_sec: float = 1.0,
    ):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * (frame_duration_ms / 1000.0))
        self.energy_threshold = energy_threshold
        self.silence_limit_frames = int(silence_duration_sec / (frame_duration_ms / 1000.0))
        self.reset()

    def reset(self) -> None:
        self.is_speech_active = False
        self.silence_counter = 0
        self.speech_frames = []

    def is_speech_frame(self, frame: np.ndarray) -> bool:
        """Calculate Root Mean Square (RMS) energy to detect speech presence."""
        if len(frame) == 0:
            return False
        rms = np.sqrt(np.mean(np.square(frame.astype(np.float32))))
        return bool(rms > self.energy_threshold)

    def process_chunk(self, chunk: np.ndarray) -> tuple[bool, bool]:
        """
        Processes an incoming audio chunk.
        Returns:
            (is_speaking, utterance_finished)
        """
        # Split chunk into frames
        utterance_finished = False

        for i in range(0, len(chunk), self.frame_size):
            frame = chunk[i : i + self.frame_size]
            if len(frame) < self.frame_size:
                continue

            speech = self.is_speech_frame(frame)

            if speech:
                self.is_speech_active = True
                self.silence_counter = 0
                self.speech_frames.append(frame)
            else:
                if self.is_speech_active:
                    self.silence_counter += 1
                    self.speech_frames.append(frame)
                    if self.silence_counter >= self.silence_limit_frames:
                        utterance_finished = True
                        self.is_speech_active = False
                        break

        return self.is_speech_active, utterance_finished

    def get_speech_audio(self) -> np.ndarray:
        """Returns the accumulated speech audio segment."""
        if not self.speech_frames:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.speech_frames)
