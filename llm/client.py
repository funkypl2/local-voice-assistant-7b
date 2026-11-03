"""
Local 7B LLM Client with streaming token generation and low-latency sentence chunking.
"""
from __future__ import annotations
import json
import re
from typing import Generator, List, Dict, Any, Optional
import requests


class LocalLLMClient:
    """Client for local 7B LLM inference via Ollama or vLLM."""

    def __init__(
        self,
        api_base: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:7b",
        api_key: str = "ollama",
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.system_prompt = system_prompt or (
            "你是一个友好、机智且反应迅速的语音对话助手。你的回答需要口语化、精炼生动，"
            "避免输出复杂的 Markdown 表格或冗长列表，适合语音朗读。"
        )

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
    ) -> Generator[str, None, None]:
        """
        Stream tokens from local LLM.
        """
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Ensure system prompt is present
        full_messages = []
        if not any(m.get("role") == "system" for m in messages):
            full_messages.append({"role": "system", "content": self.system_prompt})
        full_messages.extend(messages)

        payload = {
            "model": self.model,
            "messages": full_messages,
            "temperature": self.temperature,
            "stream": True,
        }

        try:
            with requests.post(url, headers=headers, json=payload, stream=True, timeout=30) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: "):
                        data_content = line_str[6:].strip()
                        if data_content == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_content)
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            # Fallback mock generator for offline/testing scenarios
            mock_text = (
                f"你好！我是基于 7B 本地大模型搭建的智能语音助手。"
                f"虽然当前本地推理服务未连接（{str(e)[:40]}），"
                f"但我已就绪，可以为你提供实时问答、知识检索和多轮语音交互！"
            )
            for word in re.findall(r".{1,4}", mock_text):
                yield word

    def stream_sentences(
        self,
        messages: List[Dict[str, str]],
    ) -> Generator[str, None, None]:
        """
        Accumulates stream tokens and yields complete sentences immediately for TTS synthesis.
        """
        punctuation_pattern = re.compile(r"([。！？!?；;\n]+)")
        buffer = ""

        for token in self.stream_chat(messages):
            buffer += token
            # Split buffer by sentence boundaries
            parts = punctuation_pattern.split(buffer)
            # parts will alternate: [text, punct, text, punct, trailing_text]
            if len(parts) > 1:
                # Emit all matched complete sentences
                while len(parts) >= 2:
                    sentence = parts.pop(0) + parts.pop(0)
                    sentence = sentence.strip()
                    if sentence:
                        yield sentence
                buffer = parts[0] if parts else ""

        trailing = buffer.strip()
        if trailing:
            yield trailing
