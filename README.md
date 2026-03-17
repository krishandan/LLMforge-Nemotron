# 🚀 Nemotron Web Chat (vLLM Powered)

A lightweight, extensible web chat interface built to experiment with large language models (LLMs) served via vLLM.

---

## 🎯 Purpose

This project exists to:

- Experiment with state-of-the-art open models (e.g., Nemotron, OSS models)
- Explore inference trade-offs (latency, throughput, KV cache behavior)
- Tune generation parameters (temperature, max tokens, top-k, etc.)
- Prototype RAG (Retrieval-Augmented Generation) workflows
- Build intuition for LLM system design and evaluation

---

## ✨ Features

- Simple Streamlit-based chat UI
- Connects to a vLLM inference server
- Sidebar controls (model, temperature, tokens, RAG, etc.)
- Optional RAG support
- Multi-session chat
- Built for rapid experimentation

---

## 🧱 Architecture Overview

User → Streamlit App → vLLM Server → Model  
(Optional: Vector DB → Embeddings)

---

## ⚙️ Requirements

### Hardware
- NVIDIA GPU (recommended)
- CUDA-compatible system

### Software
- Python 3.9+
- Virtual environment
- Docker (optional)

---

## 🚀 Core Components

### 1. vLLM Server (Required)

This app requires a running vLLM server.

Recommended setup:  
https://github.com/eugr/spark-vllm-docker

Example:

```bash
sudo ./run-recipe.sh nemotron-3-super-nvfp4
```

---

### 2. Run Chat App

```bash
streamlit run nemotron_chatapp.py
```

---

### 3. (Optional) Monitoring

```bash
docker compose up -d
```

Grafana: http://localhost:3001  
Prometheus: http://localhost:9091  

---

## 📊 Experimentation Areas

- TTFT, latency, throughput
- Prompt behavior
- KV cache performance
- RAG tuning

---

## 📁 Project Structure

```
Chatting_app/
├── nemotron_chatapp.py
├── requirements.txt
├── utils/
└── data/
```

---

## ⚠️ Notes

- Not production ready
- GPU dependent
- Large models require planning

---

## 💡 Why This Project

Built to explore LLM systems end-to-end and understand how model performance, UX, and system design come together in real-world AI applications.

---

## 📜 License

MIT License
