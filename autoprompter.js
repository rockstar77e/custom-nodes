"""Ollama backend — llama.cpp / GGUF inference, far faster than transformers on most GPUs.

Talks to a running Ollama server over HTTP (no extra Python deps beyond requests). Tokens
are streamed to the ComfyUI console so you can watch generation live, and `format:json`
plus a disabled "think" mode keep the output clean. Thinking is off by default.
"""

import base64
import io as _io
import json
import sys

import requests

from .caption_schema import build_user_prompt, get_system_prompt, parse_caption

DEFAULT_HOST = "http://localhost:11434"


def _host(host):
    h = (host or "").strip() or DEFAULT_HOST
    if not h.startswith("http://") and not h.startswith("https://"):
        h = "http://" + h
    return h.rstrip("/")


def list_models(host=None):
    """Return [{id, display_name}] for models pulled into the local Ollama server."""
    url = _host(host) + "/api/tags"
    try:
        r = requests.get(url, timeout=15)
    except requests.RequestException as e:
        raise ValueError(
            "Could not reach Ollama at %s. Is `ollama serve` running? (%s)" % (_host(host), e)
        )
    if r.status_code != 200:
        raise ValueError("Ollama %s: %s" % (r.status_code, r.text[:200]))
    out = []
    for m in r.json().get("models", []):
        name = m.get("name") or m.get("model")
        if name:
            out.append({"id": name, "display_name": name})
    out.sort(key=lambda x: x["id"])
    if not out:
        raise ValueError(
            "Ollama is running but has no models. Pull a vision model first, "
            "e.g. `ollama pull qwen2.5vl` or `ollama pull gemma3`."
        )
    return out


def _post_chat(url, body):
    return requests.post(url, json=body, stream=True, timeout=600)


def generate(host, model_id, idea, pil_image=None, density="normal", think=False,
             unload_after=True, stream_to_console=True):
    """Stream a caption from Ollama and return a normalized caption dict."""
    if not model_id:
        raise ValueError("No Ollama model selected. Click Fetch and pick one.")

    url = _host(host) + "/api/chat"
    user_msg = {"role": "user", "content": build_user_prompt(idea, pil_image is not None)}
    if pil_image is not None:
        buf = _io.BytesIO()
        pil_image.convert("RGB").save(buf, format="PNG")
        user_msg["images"] = [base64.b64encode(buf.getvalue()).decode("ascii")]

    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": get_system_prompt(density)},
            user_msg,
        ],
        "stream": True,
        "think": bool(think),
        "format": "json",                      # constrain output to valid JSON
        "options": {"temperature": 0.7},
        # keep_alive 0 unloads the model from VRAM the moment this response ends, instead
        # of Ollama's default 5-minute idle window. Negative would keep it resident forever.
        "keep_alive": 0 if unload_after else "5m",
    }

    try:
        r = _post_chat(url, body)
    except requests.RequestException as e:
        raise ValueError("Could not reach Ollama at %s (%s)." % (_host(host), e))

    # Some models/versions reject an explicit `think` field — retry once without it.
    if r.status_code != 200 and "think" in (r.text or "").lower():
        r.close()
        body.pop("think", None)
        r = _post_chat(url, body)

    pieces = []
    with r:
        if r.status_code != 200:
            raise ValueError("Ollama %s: %s" % (r.status_code, r.text[:300]))
        if stream_to_console:
            sys.stdout.write("\n[ideogram-autoprompter] Ollama '%s' streaming:\n" % model_id)
            sys.stdout.flush()
        for line in r.iter_lines():
            if not line:
                continue
            try:
                obj = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            if obj.get("error"):
                raise ValueError("Ollama error: %s" % obj["error"])
            msg = obj.get("message") or {}
            # surface reasoning too (when think is on) so the console shows the full picture
            for key in ("thinking", "content"):
                chunk = msg.get(key)
                if chunk:
                    if key == "content":
                        pieces.append(chunk)
                    if stream_to_console:
                        sys.stdout.write(chunk)
                        sys.stdout.flush()
            if obj.get("done"):
                break
        if stream_to_console:
            sys.stdout.write("\n")
            sys.stdout.flush()

    return parse_caption("".join(pieces))
