#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import chromadb
import httpx
from chromadb.config import Settings
from docx import Document
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ============================================================
# Config
# ============================================================

APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8080"))
VLLM_HOST = os.getenv("VLLM_HOST", "http://127.0.0.1:8000")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ============================================================
# App + state
# ============================================================

app = FastAPI(title="Nemotron Chat Workbench")

sessions: dict[str, dict[str, Any]] = {}
session_locks: dict[str, asyncio.Lock] = {}

embedder = SentenceTransformer(EMBED_MODEL_NAME)

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(anonymized_telemetry=False),
)
collection = chroma_client.get_or_create_collection(name="chatting_app_docs")

# ============================================================
# Models
# ============================================================


class SessionCreateResponse(BaseModel):
    session_id: str


class ResetRequest(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    session_id: str
    prompt: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 2048
    rag_enabled: bool = True
    tools_enabled: bool = True
    top_k: int = 4


# ============================================================
# Helpers
# ============================================================


def get_session(session_id: str) -> dict[str, Any]:
    if session_id not in sessions:
        sessions[session_id] = {
            "messages": [],
            "created_at": time.time(),
            "last_used": time.time(),
        }
    sessions[session_id]["last_used"] = time.time()
    return sessions[session_id]


def get_lock(session_id: str) -> asyncio.Lock:
    if session_id not in session_locks:
        session_locks[session_id] = asyncio.Lock()
    return session_locks[session_id]


async def fetch_models() -> list[str]:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{VLLM_HOST.rstrip('/')}/v1/models")
        r.raise_for_status()
        data = r.json()
    return [item["id"] for item in data.get("data", []) if item.get("id")]


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += max(1, chunk_size - overlap)
    return chunks


def extract_text_from_file(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix == ".docx":
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)

    raise ValueError(f"Unsupported file type: {suffix}")


def ingest_document(path: Path) -> int:
    text = extract_text_from_file(path)
    chunks = chunk_text(text)
    if not chunks:
        return 0

    embeddings = embedder.encode(chunks).tolist()
    ids = [f"{path.name}-{i}-{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
    metadatas = [{"source": path.name, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(chunks)


def retrieve_chunks(query: str, top_k: int = 4) -> list[dict[str, Any]]:
    query_embedding = embedder.encode([query]).tolist()[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    hits: list[dict[str, Any]] = []
    for doc, meta in zip(docs, metas):
        hits.append(
            {
                "text": doc,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", -1),
            }
        )
    return hits


def build_rag_context(query: str, top_k: int) -> tuple[str, list[dict[str, Any]]]:
    hits = retrieve_chunks(query, top_k=top_k)
    if not hits:
        return "", []

    parts = []
    for i, hit in enumerate(hits, start=1):
        parts.append(
            f"[Context {i} | source={hit['source']} | chunk={hit['chunk_index']}]\n{hit['text']}"
        )
    return "\n\n".join(parts), hits


# ============================================================
# Tools
# ============================================================


def tool_calculator(expression: str) -> str:
    allowed = "0123456789+-*/(). %"
    if any(ch not in allowed for ch in expression):
        return "Invalid expression"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Calculation error: {e}"


def tool_retrieve_docs(query: str, top_k: int = 4) -> str:
    hits = retrieve_chunks(query, top_k=top_k)
    if not hits:
        return "No relevant documents found."
    return "\n\n".join(
        f"Source: {hit['source']} | chunk={hit['chunk_index']}\n{hit['text']}"
        for hit in hits
    )


def tool_list_models() -> str:
    return "Use the /api/models endpoint from the UI to inspect available models."


TOOLS = {
    "calculator": tool_calculator,
    "retrieve_docs": tool_retrieve_docs,
    "list_models": tool_list_models,
}


def get_tool_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate a simple arithmetic expression.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"},
                    },
                    "required": ["expression"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "retrieve_docs",
                "description": "Search the uploaded local knowledge base for relevant context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 4},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_models",
                "description": "Explain how to inspect currently available backend models.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
    ]


def extract_assistant_message(resp: dict[str, Any]) -> dict[str, Any]:
    choices = resp.get("choices", [])
    if not choices:
        return {"content": "", "tool_calls": []}
    msg = choices[0].get("message", {}) or {}
    return {
        "content": msg.get("content", "") or "",
        "tool_calls": msg.get("tool_calls", []) or [],
    }


async def vllm_chat_once(
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    tools_enabled: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools_enabled:
        payload["tools"] = get_tool_schema()

    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(
            f"{VLLM_HOST.rstrip('/')}/v1/chat/completions",
            json=payload,
        )
        r.raise_for_status()
        return r.json()


async def run_tool_loop(
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    tools_enabled: bool,
) -> str:
    for _ in range(3):
        resp = await vllm_chat_once(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools_enabled=tools_enabled,
        )
        assistant = extract_assistant_message(resp)
        content = assistant.get("content", "")
        tool_calls = assistant.get("tool_calls", [])

        if not tool_calls:
            return content

        messages.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            }
        )

        for tc in tool_calls:
            fn = tc.get("function", {}) or {}
            name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")

            try:
                args = json.loads(raw_args) if raw_args else {}
            except Exception:
                args = {}

            tool_fn = TOOLS.get(name)
            if not tool_fn:
                result = f"Unknown tool: {name}"
            else:
                try:
                    result = tool_fn(**args)
                except Exception as e:
                    result = f"Tool execution error: {e}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "name": name,
                    "content": result,
                }
            )

    return "Tool loop limit reached without a final answer."


async def fake_stream_text(text: str):
    chunk_size = 50
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]
        yield (json.dumps({"type": "delta", "text": chunk}) + "\n").encode("utf-8")
        await asyncio.sleep(0.005)
    yield (json.dumps({"type": "done"}) + "\n").encode("utf-8")


# ============================================================
# Routes
# ============================================================


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE.replace("__VLLM_HOST__", VLLM_HOST)


@app.post("/api/session/new", response_model=SessionCreateResponse)
async def api_session_new():
    session_id = uuid.uuid4().hex
    sessions[session_id] = {
        "messages": [],
        "created_at": time.time(),
        "last_used": time.time(),
    }
    return SessionCreateResponse(session_id=session_id)


@app.get("/api/models")
async def api_models():
    try:
        models = await fetch_models()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not fetch models: {e}")
    return JSONResponse({"models": models})


@app.post("/api/reset")
async def api_reset(req: ResetRequest):
    session = get_session(req.session_id)
    session["messages"] = []
    return JSONResponse({"ok": True})


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".txt", ".pdf", ".docx"}:
        raise HTTPException(status_code=400, detail="Only .txt, .pdf, and .docx files are supported.")

    dest = UPLOAD_DIR / file.filename
    content = await file.read()
    dest.write_bytes(content)

    try:
        count = ingest_document(dest)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {e}")

    return JSONResponse({"ok": True, "filename": file.filename, "chunks_indexed": count})


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required.")

    session = get_session(req.session_id)
    lock = get_lock(req.session_id)

    async with lock:
        prior_messages = list(session["messages"])

        request_messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant running behind a local vLLM server. "
                    "Use tools only when helpful. If retrieved context is present, use it carefully "
                    "and say when the answer is not clearly supported by that context."
                ),
            }
        ]

        retrieved_hits: list[dict[str, Any]] = []
        if req.rag_enabled:
            context, retrieved_hits = build_rag_context(req.prompt, req.top_k)
            if context:
                request_messages.append(
                    {
                        "role": "system",
                        "content": f"Retrieved context:\n\n{context}",
                    }
                )

        request_messages.extend(prior_messages)
        request_messages.append({"role": "user", "content": req.prompt})

        try:
            answer = await run_tool_loop(
                model=req.model,
                messages=request_messages,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                tools_enabled=req.tools_enabled,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Chat request failed: {e}")

        session["messages"] = prior_messages + [
            {"role": "user", "content": req.prompt},
            {"role": "assistant", "content": answer},
        ]
        session["last_used"] = time.time()

        headers = {}
        if retrieved_hits:
            headers["X-RAG-HITS"] = json.dumps(retrieved_hits)[:4000]

        return StreamingResponse(
            fake_stream_text(answer),
            media_type="application/x-ndjson",
            headers=headers,
        )


# ============================================================
# Embedded UI
# ============================================================

HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Nemotron Chat Workbench</title>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/dompurify@3.2.6/dist/purify.min.js"></script>
  <style>
    :root {
      --bg: #f5efe4;
      --panel: rgba(255, 251, 244, 0.9);
      --panel-strong: #fffaf2;
      --text: #201712;
      --muted: #6f5a4f;
      --accent: #c56a2d;
      --accent-strong: #9f4f1b;
      --user: #f3d8b4;
      --assistant: #efe6d6;
      --border: rgba(111, 90, 79, 0.25);
      --shadow: 0 24px 60px rgba(58, 34, 20, 0.16);
      --error: #8f2d1d;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(197, 106, 45, 0.25), transparent 30%),
        radial-gradient(circle at bottom right, rgba(159, 79, 27, 0.18), transparent 28%),
        linear-gradient(135deg, #f8f1e4 0%, #f2e7d4 46%, #eadbc5 100%);
      display: flex;
      align-items: stretch;
      justify-content: center;
      padding: 24px;
    }

    .app {
      width: min(1200px, 100%);
      min-height: calc(100vh - 48px);
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 24px;
      overflow: hidden;
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }

    .sidebar {
      padding: 24px 20px;
      background: linear-gradient(180deg, rgba(255, 248, 238, 0.96), rgba(243, 231, 212, 0.92));
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      gap: 16px;
      overflow-y: auto;
    }

    .eyebrow {
      font-size: 12px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--muted);
    }

    .title {
      font-size: 34px;
      line-height: 0.95;
      margin: 0;
      font-weight: 700;
    }

    .subtitle {
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
      font-size: 14px;
    }

    .meta-card {
      background: rgba(255, 250, 242, 0.78);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 14px;
    }

    .meta-label {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--muted);
      margin-bottom: 8px;
    }

    .meta-value {
      font-size: 14px;
      line-height: 1.45;
      word-break: break-word;
    }

    .control-group {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .control-label {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--muted);
    }

    .control-row {
      display: grid;
      grid-template-columns: 1fr auto;
      align-items: center;
      gap: 10px;
    }

    select, input[type="number"], input[type="range"], input[type="file"], button {
      width: 100%;
      font: inherit;
    }

    select, input[type="number"], input[type="file"] {
      border-radius: 12px;
      border: 1px solid var(--border);
      background: var(--panel-strong);
      color: var(--text);
      padding: 10px 12px;
    }

    input[type="checkbox"] {
      transform: scale(1.1);
    }

    .range-wrap {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .range-value {
      font-size: 13px;
      color: var(--muted);
    }

    .side-btn {
      border: 1px solid var(--border);
      background: var(--panel-strong);
      color: var(--text);
      border-radius: 12px;
      padding: 10px 12px;
      cursor: pointer;
    }

    .side-btn.primary {
      background: linear-gradient(135deg, var(--accent), var(--accent-strong));
      color: #fff7f0;
      border: 0;
    }

    .retrieval-box {
      max-height: 180px;
      overflow-y: auto;
      white-space: pre-wrap;
      font-size: 13px;
      line-height: 1.4;
      color: var(--muted);
    }

    .main {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      min-width: 0;
      background: linear-gradient(180deg, rgba(255, 252, 247, 0.86), rgba(247, 239, 227, 0.92));
    }

    .toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 18px 22px;
      border-bottom: 1px solid var(--border);
      background: rgba(255, 249, 240, 0.75);
    }

    .status {
      color: var(--muted);
      font-size: 14px;
    }

    .toolbar-btn {
      border: 1px solid var(--border);
      background: var(--panel-strong);
      color: var(--text);
      border-radius: 999px;
      padding: 10px 14px;
      cursor: pointer;
      font: inherit;
    }

    .chat {
      padding: 24px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 16px;
      scroll-behavior: smooth;
    }

    .bubble {
      max-width: min(820px, 88%);
      padding: 16px 18px;
      border-radius: 18px;
      border: 1px solid var(--border);
      line-height: 1.65;
    }

    .bubble.user {
      align-self: flex-end;
      background: var(--user);
      border-bottom-right-radius: 6px;
    }

    .bubble.assistant {
      align-self: flex-start;
      background: var(--assistant);
      border-bottom-left-radius: 6px;
    }

    .bubble.error {
      align-self: flex-start;
      background: #f6ddd8;
      color: var(--error);
      border-color: rgba(143, 45, 29, 0.2);
      white-space: pre-wrap;
    }

    .rendered > :first-child { margin-top: 0; }
    .rendered > :last-child { margin-bottom: 0; }

    .rendered p, .rendered ul, .rendered ol, .rendered blockquote, .rendered pre, .rendered table, .rendered hr {
      margin: 0.8em 0;
    }

    .rendered pre {
      overflow-x: auto;
      padding: 14px 16px;
      border-radius: 14px;
      background: #241b18;
      color: #f8efe2;
    }

    .composer {
      padding: 20px 22px 24px;
      border-top: 1px solid var(--border);
      background: rgba(255, 249, 240, 0.82);
    }

    .controls {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 140px;
      gap: 12px;
      align-items: end;
    }

    textarea {
      width: 100%;
      resize: vertical;
      min-height: 80px;
      max-height: 220px;
      padding: 14px 16px;
      border-radius: 18px;
      border: 1px solid var(--border);
      background: var(--panel-strong);
      color: var(--text);
      font: inherit;
      line-height: 1.5;
      outline: none;
    }

    .send-btn {
      height: 52px;
      border: 0;
      border-radius: 18px;
      background: linear-gradient(135deg, var(--accent), var(--accent-strong));
      color: #fff7f0;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }

    .send-btn:disabled, .toolbar-btn:disabled, .side-btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }

    @media (max-width: 960px) {
      body { padding: 12px; }
      .app {
        min-height: calc(100vh - 24px);
        grid-template-columns: 1fr;
      }
      .sidebar {
        border-right: 0;
        border-bottom: 1px solid var(--border);
      }
      .controls { grid-template-columns: 1fr; }
      .bubble { max-width: 100%; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="eyebrow">Local vLLM chat</div>
      <h1 class="title">Nemotron<br />Workbench</h1>
      <p class="subtitle">Simple model test UI with sessions, RAG, and tool calling.</p>

      <div class="meta-card">
        <div class="meta-label">Backend</div>
        <div class="meta-value" id="backend-name">__VLLM_HOST__</div>
      </div>

      <div class="meta-card">
        <div class="meta-label">Session</div>
        <div class="meta-value" id="session-name">Not created yet</div>
      </div>

      <div class="meta-card control-group">
        <div class="meta-label">Model</div>
        <select id="model-select"></select>
      </div>

      <div class="meta-card control-group">
        <div class="meta-label">Temperature</div>
        <div class="range-wrap">
          <input id="temperature" type="range" min="0" max="2" step="0.1" value="0.2" />
          <div class="range-value" id="temperature-value">0.2</div>
        </div>
      </div>

      <div class="meta-card control-group">
        <div class="meta-label">Max tokens</div>
        <input id="max-tokens" type="number" min="1" max="32768" step="1" value="2048" />
      </div>

      <div class="meta-card control-group">
        <div class="meta-label">Top-K retrieval</div>
        <input id="top-k" type="number" min="1" max="20" step="1" value="4" />
      </div>

      <div class="meta-card control-group">
        <div class="meta-label">Options</div>
        <div class="control-row">
          <span>Enable RAG</span>
          <input id="rag-toggle" type="checkbox" checked />
        </div>
        <div class="control-row">
          <span>Enable tools</span>
          <input id="tools-toggle" type="checkbox" checked />
        </div>
      </div>

      <div class="meta-card control-group">
        <div class="meta-label">Upload knowledge file</div>
        <input id="upload-input" type="file" accept=".txt,.pdf,.docx" />
        <button class="side-btn" id="upload-btn" type="button">Upload and index</button>
      </div>

      <button class="side-btn primary" id="new-session-btn" type="button">New session</button>

      <div class="meta-card">
        <div class="meta-label">Retrieved context</div>
        <div class="retrieval-box" id="retrieval-box">No retrievals yet.</div>
      </div>
    </aside>

    <main class="main">
      <div class="toolbar">
        <div class="status" id="status">Ready to chat</div>
        <button class="toolbar-btn" id="clear-btn" type="button">Clear chat</button>
      </div>

      <section class="chat" id="chat"></section>

      <div class="composer">
        <div class="controls">
          <textarea id="prompt" placeholder="Ask the model anything..."></textarea>
          <button class="send-btn" id="send-btn" type="button">Send</button>
        </div>
      </div>
    </main>
  </div>

  <script>
    let SESSION_ID = null;

    const chatEl = document.getElementById("chat");
    const promptEl = document.getElementById("prompt");
    const sendBtn = document.getElementById("send-btn");
    const clearBtn = document.getElementById("clear-btn");
    const statusEl = document.getElementById("status");
    const modelSelect = document.getElementById("model-select");
    const sessionNameEl = document.getElementById("session-name");
    const retrievalBox = document.getElementById("retrieval-box");
    const uploadInput = document.getElementById("upload-input");
    const uploadBtn = document.getElementById("upload-btn");
    const temperatureEl = document.getElementById("temperature");
    const temperatureValueEl = document.getElementById("temperature-value");

    if (window.marked) {
      marked.setOptions({ gfm: true, breaks: true });
    }

    temperatureEl.addEventListener("input", () => {
      temperatureValueEl.textContent = temperatureEl.value;
    });

    function escapeHtml(text) {
      return text
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }

    function renderAssistantHtml(text) {
      if (!window.marked || !window.DOMPurify) {
        return `<pre>${escapeHtml(text)}</pre>`;
      }
      const html = marked.parse(text);
      return DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
    }

    function addMessage(kind, text) {
      const bubble = document.createElement("div");
      bubble.className = `bubble ${kind}`;
      if (kind === "assistant") {
        bubble.classList.add("rendered");
        bubble.innerHTML = renderAssistantHtml(text);
      } else {
        bubble.textContent = text;
      }
      chatEl.appendChild(bubble);
      chatEl.scrollTop = chatEl.scrollHeight;
      return bubble;
    }

    function updateAssistantMessage(bubble, text) {
      bubble.innerHTML = renderAssistantHtml(text);
      chatEl.scrollTop = chatEl.scrollHeight;
    }

    function setBusy(busy, label) {
      sendBtn.disabled = busy;
      clearBtn.disabled = busy;
      uploadBtn.disabled = busy;
      promptEl.disabled = busy;
      statusEl.textContent = label;
    }

    async function createSession() {
      const r = await fetch("/api/session/new", { method: "POST" });
      const data = await r.json();
      SESSION_ID = data.session_id;
      localStorage.setItem("chat_session_id", SESSION_ID);
      sessionNameEl.textContent = SESSION_ID;
      retrievalBox.textContent = "No retrievals yet.";
      chatEl.innerHTML = "";
      addMessage("assistant", "New session created. You can start testing the model.");
    }

    async function ensureSession() {
      const existing = localStorage.getItem("chat_session_id");
      if (existing) {
        SESSION_ID = existing;
        sessionNameEl.textContent = SESSION_ID;
      } else {
        await createSession();
      }
    }

    async function loadModels() {
      const r = await fetch("/api/models");
      const data = await r.json();
      modelSelect.innerHTML = "";
      const models = data.models || [];
      for (const model of models) {
        const opt = document.createElement("option");
        opt.value = model;
        opt.textContent = model;
        modelSelect.appendChild(opt);
      }
    }

    async function clearChat() {
      if (!SESSION_ID) return;
      await fetch("/api/reset", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ session_id: SESSION_ID })
      });
      chatEl.innerHTML = "";
      retrievalBox.textContent = "No retrievals yet.";
      addMessage("assistant", "Conversation cleared. What would you like to test next?");
    }

    async function uploadDoc() {
      const file = uploadInput.files[0];
      if (!file) {
        addMessage("error", "Choose a .txt, .pdf, or .docx file first.");
        return;
      }

      const fd = new FormData();
      fd.append("file", file);

      setBusy(true, "Uploading and indexing...");
      try {
        const r = await fetch("/api/upload", {
          method: "POST",
          body: fd
        });
        const data = await r.json();
        if (!r.ok) {
          throw new Error(data.detail || "Upload failed");
        }
        addMessage("assistant", `Indexed file: ${data.filename} (${data.chunks_indexed} chunks).`);
        uploadInput.value = "";
      } catch (err) {
        addMessage("error", String(err));
      } finally {
        setBusy(false, "Ready to chat");
      }
    }

    async function sendMessage() {
      const prompt = promptEl.value.trim();
      if (!prompt) return;
      if (!SESSION_ID) {
        addMessage("error", "Session not ready.");
        return;
      }

      const model = modelSelect.value;
      const temperature = parseFloat(document.getElementById("temperature").value);
      const max_tokens = parseInt(document.getElementById("max-tokens").value, 10);
      const top_k = parseInt(document.getElementById("top-k").value, 10);
      const rag_enabled = document.getElementById("rag-toggle").checked;
      const tools_enabled = document.getElementById("tools-toggle").checked;

      addMessage("user", prompt);
      promptEl.value = "";
      setBusy(true, "Thinking...");

      const assistantBubble = addMessage("assistant", "");
      let fullText = "";

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: SESSION_ID,
            prompt,
            model,
            temperature,
            max_tokens,
            rag_enabled,
            tools_enabled,
            top_k
          })
        });

        if (!response.ok) {
          const data = await response.json();
          assistantBubble.remove();
          addMessage("error", data.detail || `HTTP ${response.status}`);
          return;
        }

        const ragHeader = response.headers.get("X-RAG-HITS");
        if (ragHeader) {
          try {
            const hits = JSON.parse(ragHeader);
            retrievalBox.textContent = hits.map(
              (h, i) => `${i + 1}. ${h.source} [chunk ${h.chunk_index}]\n${h.text}`
            ).join("\n\n");
          } catch (_) {
          }
        } else {
          retrievalBox.textContent = "No retrievals for this prompt.";
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          while (buffer.includes("\n")) {
            const idx = buffer.indexOf("\n");
            const line = buffer.slice(0, idx).trim();
            buffer = buffer.slice(idx + 1);
            if (!line) continue;

            let payload;
            try {
              payload = JSON.parse(line);
            } catch (_) {
              continue;
            }

            if (payload.type === "delta") {
              fullText += payload.text || "";
              updateAssistantMessage(assistantBubble, fullText);
            } else if (payload.type === "done") {
              updateAssistantMessage(assistantBubble, fullText || "(empty response)");
            }
          }
        }

        updateAssistantMessage(assistantBubble, fullText || "(empty response)");
      } catch (err) {
        assistantBubble.remove();
        addMessage("error", String(err));
      } finally {
        setBusy(false, "Ready to chat");
        promptEl.focus();
      }
    }

    sendBtn.addEventListener("click", sendMessage);
    clearBtn.addEventListener("click", clearChat);
    uploadBtn.addEventListener("click", uploadDoc);
    document.getElementById("new-session-btn").addEventListener("click", createSession);

    promptEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
      }
    });

    window.addEventListener("DOMContentLoaded", async () => {
      await ensureSession();
      await loadModels();
      addMessage("assistant", "Connected. Choose a model in the sidebar and send a prompt when ready.");
      promptEl.focus();
    });
  </script>
</body>
</html>"""
