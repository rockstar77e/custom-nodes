# ComfyUI Ideogram 4 Autoprompter

An AI autoprompter node for Ideogram 4's structured JSON caption format. Describe an
idea (and/or drop a reference image) in the node, click **Generate**, and a vision LLM
builds the full caption — background, placed elements with bounding boxes, descriptions,
rendered text, and color palettes. Everything stays editable on a visual black-and-white
canvas afterward. At workflow run-time the node only passes the assembled caption JSON
string through (plus a preview image).

## Install

Copy this folder into `ComfyUI/custom_nodes/` and install requirements into the ComfyUI
Python environment:

```
pip install -r requirements.txt
```

`torch` is provided by ComfyUI. For the optional **4-bit** local-model toggle, also
`pip install bitsandbytes` (CUDA only); without it the node simply loads in full precision.

## Engines

- **Local (default):** [`huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated`](https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated),
  downloaded automatically on first generate via `transformers`. The model is **unloaded
  after each generation by default** (toggle "unload after"). You can point the model-id
  field at any compatible Qwen3-VL checkpoint.
- **Gemini (free API key):** paste a key, click **Fetch models**, pick one, then Generate.
  The API key is kept in memory for the session only — it is never saved into the workflow.

## Usage

1. Add the **Ideogram 4 Autoprompter** node (category `Ideogram/text`).
2. Pick an engine, type an idea and/or drop a reference image, click **Generate**.
3. The canvas and fields populate from the generated caption. Edit freely:
   draw/resize/move regions, set type (obj/text), descriptions, text, and palettes.
4. The `prompt` output is the caption JSON string; `preview` is a rendered layout image.

Generation happens on the node UI (via the button), not during graph execution.
