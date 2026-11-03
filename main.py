"""
Interactive CLI for Local-VoiceAssistant-7B.
"""
from __future__ import annotations
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import argparse
from core.assistant import VoiceAssistant
from llm.client import LocalLLMClient
from asr.engine import FasterWhisperASR
from tts.engine import TTSEngine
from audio.io import AudioIO


def main():
    parser = argparse.ArgumentParser(description="Local 7B Voice Assistant CLI")
    parser.add_argument("--api-base", type=str, default="http://localhost:11434/v1", help="Ollama or vLLM API base URL")
    parser.add_argument("--model", type=str, default="qwen2.5:7b", help="Model name (e.g. qwen2.5:7b, llama3:8b)")
    parser.add_argument("--voice", type=str, default="zh-CN-XiaoxiaoNeural", help="Edge-TTS voice")
    parser.add_argument("--mode", type=str, default="text", choices=["text", "voice", "demo"], help="Interaction mode")
    args = parser.parse_args()

    llm = LocalLLMClient(api_base=args.api_base, model=args.model)
    asr = FasterWhisperASR()
    tts = TTSEngine(voice=args.voice)
    audio = AudioIO()

    assistant = VoiceAssistant(llm_client=llm, asr_engine=asr, tts_engine=tts, audio_io=audio)

    print("==========================================================")
    print("      Local-VoiceAssistant-7B (Qwen2.5 / vLLM / ASR)      ")
    print(f"  Model: {args.model} | Mode: {args.mode}")
    print("==========================================================")

    if args.mode == "demo":
        print("[*] Running automated end-to-end demonstration...")
        u_text, a_reply = assistant.step(text_input="请用简短一句话介绍一下你自己和你的能力。")
        print(f"[User]: {u_text}")
        print(f"[Assistant]: {a_reply}")
        return

    if args.mode == "voice":
        print("[*] Voice mode: Press Enter to record 4 seconds of audio (Ctrl+C to exit)...")
        while True:
            try:
                input("\n[Press ENTER and speak] >>> ")
                print("[*] Recording audio...")
                rec_audio = audio.record_seconds(duration=4.0)
                print("[*] Processing ASR & LLM...")
                u_text, a_reply = assistant.step(user_audio=rec_audio)
                print(f"[Transcribed]: {u_text}")
                print(f"[Assistant]: {a_reply}")
            except KeyboardInterrupt:
                print("\nExiting.")
                break
    else:
        print("[*] Text simulation mode: Type your message (Ctrl+C or 'exit' to quit)...")
        while True:
            try:
                query = input("\n[You] >>> ").strip()
                if not query or query.lower() in ("exit", "quit"):
                    break
                u_text, a_reply = assistant.step(text_input=query)
                print(f"[Assistant]: {a_reply}")
            except KeyboardInterrupt:
                print("\nExiting.")
                break


if __name__ == "__main__":
    main()
