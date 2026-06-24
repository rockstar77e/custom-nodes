"""Gemini backend over the public Generative Language REST API.

Kept to plain `requests` so no extra SDK is required. The user supplies a free API
key in the node UI, fetches the available models, then generates. The key is never
written into the workflow.
"""

import base64
import io as _io

import requests

from .caption_schema import build_user_prompt, get_system_prompt, parse_caption

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"


def list_models(api_key):
    """Return [{id, display_name}] for models that support generateContent."""
    if not api_key:
        raise ValueError("No API key provided.")
    r = requests.get("%s/models" % API_ROOT, params={"key": api_key}, timeout=30)
    if r.status_code != 200:
        raise ValueError(_err(r))
    out = []
    for m in r.json().get("models", []):
        if "generateContent" in (m.get("supportedGenerationMethods") or []):
            name = m.get("name", "")              # "models/gemini-2.0-flash"
            mid = name.split("/", 1)[-1] if "/" in name else name
            out.append({"id": mid, "display_name": m.get("displayName", mid)})
    out.sort(key=lambda x: x["id"])
    return out


def _err(resp):
    try:
        e = resp.json().get("error", {})
        return "Gemini API %s: %s" % (resp.status_code, e.get("message", resp.text[:200]))
    except Exception:
        return "Gemini API %s: %s" % (resp.status_code, resp.text[:200])


def generate(api_key, model_id, idea, pil_image=None, density="normal"):
    """Call generateContent and return a normalized caption dict."""
    if not api_key:
        raise ValueError("No API key provided.")
    if not model_id:
        raise ValueError("No model selected.")

    parts = [{"text": build_user_prompt(idea, pil_image is not None)}]
    if pil_image is not None:
        buf = _io.BytesIO()
        pil_image.convert("RGB").save(buf, format="PNG")
        parts.append({
            "inline_data": {
                "mime_type": "image/png",
                "data": base64.b64encode(buf.getvalue()).decode("ascii"),
            }
        })

    body = {
        "systemInstruction": {"parts": [{"text": get_system_prompt(density)}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.7,
            "responseMimeType": "application/json",
        },
    }
    url = "%s/models/%s:generateContent" % (API_ROOT, model_id)
    r = requests.post(url, params={"key": api_key}, json=body, timeout=120)
    if r.status_code != 200:
        raise ValueError(_err(r))

    data = r.json()
    candidates = data.get("candidates") or []
    if not candidates:
        fb = data.get("promptFeedback", {})
        raise ValueError("Gemini returned no candidates (%s)." % fb.get("blockReason", "unknown"))
    text = "".join(
        p.get("text", "") for p in candidates[0].get("content", {}).get("parts", [])
    )
    return parse_caption(text)
