# Local-VoiceAssistant-7B: Edge AI Voice Companion

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Model: Qwen2.5-7B](https://img.shields.io/badge/LLM-Qwen2.5--7B%20%2F%20Llama3--8B-purple.svg)]()
[![ASR: Faster-Whisper](https://img.shields.io/badge/ASR-Faster--Whisper-orange.svg)]()
[![TTS: Edge-TTS](https://img.shields.io/badge/TTS-Edge--TTS%20%2F%20Local-red.svg)]()

**Local-VoiceAssistant-7B** 是一个完全运行在本地的端到端智能语音对话助手。基于开源 7B 大语言模型（如 `Qwen2.5-7B-Instruct`、`Llama-3-8B`），结合 **Faster-Whisper 本地语音识别**、**Silero/Energy 双模 VAD 静音切片** 与 **流式分句合成（Streaming Sentence Chunking TTS）** 技术，将首字发声延迟（Time-To-First-Audio, TTFA）压缩至 300ms 以内。

---

## 🏗️ 架构流向图

```
 [Microphone] ──► [VAD 语音端点检测] ──► [Faster-Whisper ASR]
                                                   │ (转写文本)
                                                   ▼
                                        [7B Local LLM 推理引擎]
                                        (Ollama / vLLM / GGUF)
                                                   │
                                                   ▼ (流式 Token 生成)
                                        [Sentence Splitter 分句器]
                                                   │ (标点断句流)
                                                   ▼
 [Speaker] ◄── [Audio Queue 播放缓冲] ◄── [Edge-TTS / 本地合成]
```

### 核心亮点
1. **纯本地隐私保护**：支持使用 Ollama 或 vLLM 完全离线部署 7B 模型，数据不出本地内网。
2. **极低交互延迟**：通过流式 Token 动态断句算法，不必等待 LLM 全部生成完毕，首句生成即刻送入 TTS 合成发声。
3. **软硬件解耦与平滑降级**：内置自动化 Mock 与容错通道，在缺乏独立 GPU 或外部音频设备时自动回退，确保任何开发环境都能正常运行并测试。

---

## 🚀 快速上手 (Quick Start)

### 1. 安装依赖
```bash
cd 02-local-voice-assistant-7b
pip install -r requirements.txt
```

### 2. 启动本地 7B 大模型服务
推荐使用 **Ollama**（一键跨平台安装）：
```bash
# 启动 Qwen2.5-7B-Instruct
ollama run qwen2.5:7b
```
或者使用 **vLLM**（高并发高性能推理）：
```bash
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --port 11434
```

### 3. 运行语音助手

**自动化演示模式（无需麦克风）：**
```bash
python main.py --mode demo
```

**文字模拟调试模式：**
```bash
python main.py --mode text --api-base http://localhost:11434/v1 --model qwen2.5:7b
```

**真实麦克风交互模式：**
```bash
python main.py --mode voice --api-base http://localhost:11434/v1 --model qwen2.5:7b
```

### 4. 运行单元测试
```bash
pytest tests/ -v
```

---

## ⚙️ 配置说明

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--api-base` | `http://localhost:11434/v1` | OpenAI 兼容的本地 API 地址 |
| `--model` | `qwen2.5:7b` | 模型代号（支持 qwen2.5:7b, llama3:8b, mistral 等） |
| `--voice` | `zh-CN-XiaoxiaoNeural` | TTS 发音人声音类型 |
| `--mode` | `text` | 运行模式：`demo`、`text`、`voice` |

---

## 📄 开源许可证
本项目采用 [MIT License](LICENSE) 授权协议。
