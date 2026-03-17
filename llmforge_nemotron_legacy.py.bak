#!/usr/bin/env python3
"""
Local browser chat app for a vLLM OpenAI-compatible server.

Run:
  python3 llmforge_nemotron_legacy.py

Then open:
  http://127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
DEFAULT_VLLM_HOST = "http://127.0.0.1:8000"

HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LLMforge-Nemotron</title>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/dompurify@3.2.6/dist/purify.min.js"></script>
  <script type="module">
    import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
    window.mermaid = mermaid;
    mermaid.initialize({ startOnLoad: false, theme: "neutral", securityLevel: "loose" });
  </script>
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

    * {
      box-sizing: border-box;
    }

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
      width: min(1080px, 100%);
      min-height: calc(100vh - 48px);
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr);
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 24px;
      overflow: hidden;
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }

    .sidebar {
      padding: 28px 22px;
      background: linear-gradient(180deg, rgba(255, 248, 238, 0.96), rgba(243, 231, 212, 0.92));
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      gap: 18px;
    }

    .eyebrow {
      font-size: 12px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--muted);
    }

    .title {
      font-size: clamp(30px, 5vw, 44px);
      line-height: 0.95;
      margin: 0;
      font-weight: 700;
    }

    .subtitle {
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
    }

    .meta-card {
      background: rgba(255, 250, 242, 0.78);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 16px;
    }

    .meta-label {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--muted);
      margin-bottom: 6px;
    }

    .meta-value {
      font-size: 15px;
      line-height: 1.45;
      word-break: break-word;
    }

    .tips {
      margin: auto 0 0;
      padding-left: 18px;
      color: var(--muted);
      line-height: 1.55;
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

    .clear-btn {
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
      animation: rise 180ms ease-out;
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

    .rendered > :first-child {
      margin-top: 0;
    }

    .rendered > :last-child {
      margin-bottom: 0;
    }

    .rendered p,
    .rendered ul,
    .rendered ol,
    .rendered blockquote,
    .rendered pre,
    .rendered table,
    .rendered hr {
      margin: 0.8em 0;
    }

    .rendered ul,
    .rendered ol {
      padding-left: 1.4em;
    }

    .rendered li + li {
      margin-top: 0.25em;
    }

    .rendered h1,
    .rendered h2,
    .rendered h3,
    .rendered h4 {
      margin: 0.9em 0 0.4em;
      line-height: 1.2;
    }

    .rendered h1 {
      font-size: 1.5rem;
    }

    .rendered h2 {
      font-size: 1.3rem;
    }

    .rendered h3 {
      font-size: 1.15rem;
    }

    .rendered blockquote {
      padding: 0.2em 0 0.2em 1em;
      border-left: 4px solid rgba(197, 106, 45, 0.55);
      color: #59463c;
      background: rgba(255, 250, 242, 0.55);
      border-radius: 0 12px 12px 0;
    }

    .rendered code,
    .rendered pre {
      font-family: "Iosevka Term", "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 0.92em;
    }

    .rendered :not(pre) > code {
      padding: 0.15em 0.45em;
      border-radius: 8px;
      background: rgba(255, 249, 240, 0.95);
      border: 1px solid rgba(111, 90, 79, 0.18);
    }

    .rendered pre {
      overflow-x: auto;
      padding: 14px 16px;
      border-radius: 14px;
      background: #241b18;
      color: #f8efe2;
      border: 1px solid rgba(111, 90, 79, 0.18);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }

    .rendered pre code {
      background: transparent;
      border: 0;
      padding: 0;
      color: inherit;
      white-space: pre;
    }

    .table-wrap {
      overflow-x: auto;
      border: 1px solid rgba(111, 90, 79, 0.18);
      border-radius: 14px;
      background: rgba(255, 250, 242, 0.75);
    }

    .rendered table {
      width: 100%;
      min-width: 540px;
      border-collapse: collapse;
      margin: 0;
      font-size: 0.96em;
    }

    .rendered th,
    .rendered td {
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
      border-bottom: 1px solid rgba(111, 90, 79, 0.15);
    }

    .rendered th {
      position: sticky;
      top: 0;
      background: #ead7bc;
      font-weight: 700;
    }

    .rendered tr:nth-child(even) td {
      background: rgba(255, 255, 255, 0.35);
    }

    .rendered hr {
      border: 0;
      border-top: 1px solid rgba(111, 90, 79, 0.2);
    }

    .diagram-wrap {
      overflow-x: auto;
      padding: 14px;
      border-radius: 14px;
      background: rgba(255, 250, 242, 0.82);
      border: 1px solid rgba(111, 90, 79, 0.18);
    }

    .diagram-wrap svg {
      display: block;
      max-width: 100%;
      height: auto;
      margin: 0 auto;
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
      min-height: 74px;
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

    textarea:focus,
    .clear-btn:focus,
    .send-btn:focus {
      border-color: rgba(197, 106, 45, 0.6);
      box-shadow: 0 0 0 3px rgba(197, 106, 45, 0.12);
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
      box-shadow: 0 10px 30px rgba(159, 79, 27, 0.18);
    }

    .send-btn:disabled,
    .clear-btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      box-shadow: none;
    }

    @keyframes rise {
      from {
        opacity: 0;
        transform: translateY(6px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    @media (max-width: 860px) {
      body {
        padding: 12px;
      }

      .app {
        min-height: calc(100vh - 24px);
        grid-template-columns: 1fr;
      }

      .sidebar {
        gap: 12px;
        border-right: 0;
        border-bottom: 1px solid var(--border);
      }

      .tips {
        margin-top: 0;
      }

      .controls {
        grid-template-columns: 1fr;
      }

      .send-btn {
        width: 100%;
      }

      .bubble {
        max-width: 100%;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="eyebrow">Local vLLM chat</div>
      <h1 class="title">LLMforge<br />Nemotron</h1>
      <p class="subtitle">A focused workspace for your running vLLM Nemotron server.</p>

      <div class="meta-card">
        <div class="meta-label">Model</div>
        <div class="meta-value" id="model-name"></div>
      </div>

      <div class="meta-card">
        <div class="meta-label">Endpoint</div>
        <div class="meta-value" id="endpoint-name"></div>
      </div>

      <ul class="tips">
        <li>Press Enter to send.</li>
        <li>Use Shift+Enter for a new line.</li>
        <li>The conversation stays in memory until you clear it or restart the app.</li>
      </ul>
    </aside>

    <main class="main">
      <div class="toolbar">
        <div class="status" id="status">Ready to chat</div>
        <button class="clear-btn" id="clear-btn" type="button">Clear chat</button>
      </div>

      <section class="chat" id="chat"></section>

      <div class="composer">
        <div class="controls">
          <textarea id="prompt" placeholder="Ask Nemotron anything..."></textarea>
          <button class="send-btn" id="send-btn" type="button">Send</button>
        </div>
      </div>
    </main>
  </div>

  <script>
    const MODEL = "__MODEL__";
    const ENDPOINT = "__ENDPOINT__";
    const chatEl = document.getElementById("chat");
    const promptEl = document.getElementById("prompt");
    const sendBtn = document.getElementById("send-btn");
    const clearBtn = document.getElementById("clear-btn");
    const statusEl = document.getElementById("status");

    document.getElementById("model-name").textContent = MODEL;
    document.getElementById("endpoint-name").textContent = ENDPOINT;

    if (window.marked) {
      marked.setOptions({
        gfm: true,
        breaks: true
      });
    }

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

    async function enhanceRichContent(container) {
      container.querySelectorAll("table").forEach((table) => {
        if (table.parentElement && table.parentElement.classList.contains("table-wrap")) {
          return;
        }
        const wrapper = document.createElement("div");
        wrapper.className = "table-wrap";
        table.parentNode.insertBefore(wrapper, table);
        wrapper.appendChild(table);
      });

      const mermaidBlocks = Array.from(container.querySelectorAll("pre > code.language-mermaid, pre > code.lang-mermaid"));
      for (const block of mermaidBlocks) {
        const source = block.textContent || "";
        const wrapper = document.createElement("div");
        wrapper.className = "diagram-wrap";
        const id = `mermaid-${Date.now()}-${Math.random().toString(36).slice(2)}`;
        try {
          if (window.mermaid) {
            const { svg } = await window.mermaid.render(id, source);
            wrapper.innerHTML = svg;
          } else {
            wrapper.innerHTML = `<pre>${escapeHtml(source)}</pre>`;
          }
        } catch (error) {
          wrapper.innerHTML = `<pre>${escapeHtml(source)}</pre>`;
        }
        const pre = block.parentElement;
        if (pre && pre.parentElement) {
          pre.parentElement.replaceChild(wrapper, pre);
        }
      }
    }

    function addMessage(kind, text) {
      const bubble = document.createElement("div");
      bubble.className = `bubble ${kind}`;
      if (kind === "assistant") {
        bubble.classList.add("rendered");
        bubble.innerHTML = renderAssistantHtml(text);
        enhanceRichContent(bubble);
      } else {
        bubble.textContent = text;
      }
      chatEl.appendChild(bubble);
      chatEl.scrollTop = chatEl.scrollHeight;
      return bubble;
    }

    function updateAssistantMessage(bubble, text) {
      bubble.innerHTML = renderAssistantHtml(text);
      enhanceRichContent(bubble);
      chatEl.scrollTop = chatEl.scrollHeight;
    }

    function setBusy(busy, label) {
      sendBtn.disabled = busy;
      clearBtn.disabled = busy;
      promptEl.disabled = busy;
      statusEl.textContent = label;
    }

    async function sendMessage() {
      const prompt = promptEl.value.trim();
      if (!prompt) return;

      addMessage("user", prompt);
      promptEl.value = "";
      setBusy(true, "Thinking...");

      const assistantBubble = addMessage("assistant", "");
      let fullText = "";

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt })
        });
        if (!response.ok) {
          const data = await response.json();
          assistantBubble.remove();
          addMessage("error", data.error || `HTTP ${response.status}`);
        } else if (!response.body) {
          const data = await response.json();
          updateAssistantMessage(assistantBubble, data.response || "(empty response)");
        } else {
          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer = "";

          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            while (buffer.includes("\\n")) {
              const newlineIndex = buffer.indexOf("\\n");
              const line = buffer.slice(0, newlineIndex).trim();
              buffer = buffer.slice(newlineIndex + 1);

              if (!line) continue;
              let payload;
              try {
                payload = JSON.parse(line);
              } catch (error) {
                continue;
              }

              if (payload.type === "delta") {
                fullText += payload.text || "";
                updateAssistantMessage(assistantBubble, fullText);
              } else if (payload.type === "done") {
                updateAssistantMessage(assistantBubble, fullText || "(empty response)");
              } else if (payload.type === "error") {
                assistantBubble.remove();
                addMessage("error", payload.error || "Streaming failed");
                return;
              }
            }
          }

          if (buffer.trim()) {
            try {
              const payload = JSON.parse(buffer.trim());
              if (payload.type === "delta") {
                fullText += payload.text || "";
              }
            } catch (error) {
            }
          }

          updateAssistantMessage(assistantBubble, fullText || "(empty response)");
        }
      } catch (error) {
        assistantBubble.remove();
        addMessage("error", String(error));
      } finally {
        setBusy(false, "Ready to chat");
        promptEl.focus();
      }
    }

    async function clearChat() {
      try {
        await fetch("/api/reset", { method: "POST" });
      } catch (error) {
        addMessage("error", String(error));
      }
      chatEl.innerHTML = "";
      addMessage("assistant", "Conversation cleared. What would you like to ask next?");
      promptEl.focus();
    }

    sendBtn.addEventListener("click", sendMessage);
    clearBtn.addEventListener("click", clearChat);
    promptEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
      }
    });

    addMessage("assistant", "Connected to Nemotron. Send a prompt when you're ready.");
    promptEl.focus();
  </script>
</body>
</html>
"""


def extract_message_content(choice: dict) -> str:
    message = choice.get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content).strip()


def vllm_chat(
    *,
    vllm_host: str,
    model: str,
    messages: list[dict[str, str]],
    api_key: str | None,
    timeout: int,
    temperature: float,
    max_tokens: int | None,
) -> str:
    payload: dict[str, object] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
      url=f"{vllm_host.rstrip('/')}/v1/chat/completions",
      data=json.dumps(payload).encode("utf-8"),
      headers=headers,
      method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))

    choices = data.get("choices") or []
    if not choices:
        return ""
    return extract_message_content(choices[0])


def vllm_chat_stream(
    *,
    vllm_host: str,
    model: str,
    messages: list[dict[str, str]],
    api_key: str | None,
    timeout: int,
    temperature: float,
    max_tokens: int | None,
):
    payload: dict[str, object] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        url=f"{vllm_host.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                payload = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            choices = payload.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content", "")
            if isinstance(content, str) and content:
                yield content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text")
                        if isinstance(text, str) and text:
                            yield text


def ensure_vllm_up(vllm_host: str, api_key: str | None) -> None:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url=f"{vllm_host.rstrip('/')}/v1/models",
        headers=headers,
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=5):
        return


def build_handler(
    *,
    model: str,
    vllm_host: str,
    api_key: str | None,
    system_prompt: str | None,
    timeout: int,
    temperature: float,
    max_tokens: int | None,
):
    conversations: dict[str, list[dict[str, str]]] = {"default": []}

    class Handler(BaseHTTPRequestHandler):
        def json_response(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def stream_json_line(self, payload: dict) -> None:
            body = (json.dumps(payload) + "\n").encode("utf-8")
            self.wfile.write(body)
            self.wfile.flush()

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/":
                self.send_error(404, "Not Found")
                return

            page = (
                HTML.replace("__MODEL__", model)
                .replace("__ENDPOINT__", vllm_host)
                .encode("utf-8")
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/api/reset":
                conversations["default"] = []
                self.json_response(200, {"ok": True})
                return

            if self.path != "/api/chat":
                self.send_error(404, "Not Found")
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length else b""
                payload = json.loads(body.decode("utf-8")) if body else {}
                prompt = str(payload.get("prompt", "")).strip()
                if not prompt:
                    self.json_response(400, {"error": "prompt is required"})
                    return

                conversation = list(conversations["default"])
                request_messages: list[dict[str, str]] = []
                if system_prompt:
                    request_messages.append({"role": "system", "content": system_prompt})
                request_messages.extend(conversation)
                request_messages.append({"role": "user", "content": prompt})

                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                full_answer = []
                for chunk in vllm_chat_stream(
                    vllm_host=vllm_host,
                    model=model,
                    messages=request_messages,
                    api_key=api_key,
                    timeout=timeout,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    full_answer.append(chunk)
                    self.stream_json_line({"type": "delta", "text": chunk})

                answer = "".join(full_answer)
                conversations["default"] = conversation + [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ]
                self.stream_json_line({"type": "done"})
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")
                if not self.wfile.closed:
                    try:
                        self.stream_json_line({"type": "error", "error": f"vLLM HTTP {error.code}: {detail}"})
                    except Exception:
                        self.json_response(502, {"error": f"vLLM HTTP {error.code}: {detail}"})
            except Exception as error:  # noqa: BLE001
                if not self.wfile.closed:
                    try:
                        self.stream_json_line({"type": "error", "error": str(error)})
                    except Exception:
                        self.json_response(500, {"error": str(error)})

        def log_message(self, fmt: str, *args) -> None:
            return

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local chat UI backed by vLLM.")
    parser.add_argument("--host", default="127.0.0.1", help="Web app host.")
    parser.add_argument("--port", type=int, default=8080, help="Web app port.")
    parser.add_argument("--vllm-host", default=DEFAULT_VLLM_HOST, help="vLLM API base URL.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name exposed by vLLM.")
    parser.add_argument("--api-key", default=None, help="Optional API key for the vLLM server.")
    parser.add_argument("--timeout", type=int, default=600, help="vLLM request timeout in seconds.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=50000, help="Maximum completion tokens.")
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful AI assistant running on NVIDIA Nemotron via vLLM.",
        help="Optional system prompt.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        ensure_vllm_up(args.vllm_host, args.api_key)
    except Exception as error:  # noqa: BLE001
        print(f"[ERROR] Could not reach vLLM at {args.vllm_host}: {error}")
        return 1

    handler = build_handler(
        model=args.model,
        vllm_host=args.vllm_host,
        api_key=args.api_key,
        system_prompt=args.system_prompt,
        timeout=args.timeout,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print(f"[INFO] Web UI: http://{args.host}:{args.port}")
    print(f"[INFO] vLLM API: {args.vllm_host}")
    print(f"[INFO] Model: {args.model}")
    print("[INFO] Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
