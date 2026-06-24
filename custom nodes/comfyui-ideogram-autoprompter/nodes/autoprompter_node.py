"""Ideogram 4 Autoprompter node.

Same runtime contract as the KJ Ideogram 4 Prompt Builder: the editor state is
serialized into hidden widgets and assembled into a strict-key-order caption JSON at
graph-run time (plus a rendered preview). The AI layer (idea/image -> caption) runs
entirely from the node UI through the HTTP routes; it never touches execute().
"""

import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from comfy_api.latest import io

_FONT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fonts", "FreeMono.ttf")


def _hex_rgb(h):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)) if len(h) == 6 else (255, 255, 255)


def _font(size):
    try:
        return ImageFont.truetype(_FONT_PATH, size)
    except Exception:
        try:
            return ImageFont.load_default(size)
        except Exception:
            return ImageFont.load_default()


def _wrap(draw, text, font, max_w):
    lines = []
    for para in text.split("\n"):
        line = ""
        for word in para.split():
            test = word if not line else line + " " + word
            if line and draw.textlength(test, font=font) > max_w:
                lines.append(line)
                line = word
            else:
                line = test
        lines.append(line)
    return lines


def _render_preview(boxes, width, height):
    # Render the regions + prompts on a black canvas, mirroring the JS editor.
    long_edge = max(width, height)
    scale = min(1.0, 1024 / long_edge) if long_edge > 0 else 1.0
    rw = max(1, round(width * scale))
    rh = max(1, round(height * scale))

    img = Image.new("RGBA", (rw, rh), (0, 0, 0, 255))
    overlay = Image.new("RGBA", (rw, rh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    fs = max(10, round(rh / 64))
    font = _font(fs)
    tag_font = _font(max(9, fs - 2))
    lh = fs + 2

    for i, box in enumerate(boxes):
        if not isinstance(box, dict) or box.get("nobbox"):
            continue
        palette = [c for c in (box.get("palette") or []) if c]
        r, g, b = _hex_rgb(palette[0]) if palette else (140, 140, 140)
        x1 = max(0, min(rw, round(box.get("x", 0) * rw)))
        y1 = max(0, min(rh, round(box.get("y", 0) * rh)))
        x2 = max(0, min(rw, round((box.get("x", 0) + box.get("w", 0)) * rw)))
        y2 = max(0, min(rh, round((box.get("y", 0) + box.get("h", 0)) * rh)))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        draw.rectangle([x1, y1, x2, y2], outline=(r, g, b, 255), width=2)

        pal5 = palette[:5]
        if pal5 and (x2 - x1) > 2:
            sh = max(5, fs // 2)
            seg = (x2 - x1) / len(pal5)
            for p, hexc in enumerate(pal5):
                sx = x1 + round(p * seg)
                draw.rectangle([sx, y1, x1 + round((p + 1) * seg), y1 + sh], fill=_hex_rgb(hexc))

        etype = "text" if box.get("type") == "text" else "obj"
        tag = str(i + 1).zfill(2)
        tw = draw.textlength(tag, font=tag_font)
        draw.rectangle([x1, y1, x1 + tw + 6, y1 + fs + 2], fill=(r, g, b, 255))
        tagfill = (0, 0, 0, 255) if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else (255, 255, 255, 255)
        draw.text((x1 + 3, y1 + 1), tag, fill=tagfill, font=tag_font)

        body = box.get("desc", "") or ""
        if etype == "text" and box.get("text"):
            body = '"%s"%s' % (box["text"], " — " + body if body else "")
        if body and (x2 - x1) > 8:
            ty = y1 + fs + 5
            for line in _wrap(draw, body, font, x2 - x1 - 8):
                if ty > y2:
                    break
                draw.text((x1 + 4, ty), line, fill=(212, 212, 212, 255), font=font)
                ty += lh

    img = Image.alpha_composite(img, overlay).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _norm_bbox(box):
    def c(v):
        return max(0, min(1000, round(v * 1000)))
    x, y, w, h = box.get("x", 0.0), box.get("y", 0.0), box.get("w", 0.0), box.get("h", 0.0)
    ymin, xmin, ymax, xmax = c(y), c(x), c(y + h), c(x + w)
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    return [ymin, xmin, ymax, xmax]


def _palette(colors):
    if isinstance(colors, dict):
        colors = colors.values()
    return [c.upper() for c in colors if c]


def _dumps(v, lvl=0):
    # Like json.dumps(ensure_ascii=False, indent=4), but scalar arrays stay on one line.
    pad, end = "    " * (lvl + 1), "    " * lvl
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        if not v:
            return "[]"
        if all(not isinstance(x, (dict, list)) for x in v):
            return "[" + ", ".join(_dumps(x, lvl) for x in v) + "]"
        return "[\n" + ",\n".join(pad + _dumps(x, lvl + 1) for x in v) + "\n" + end + "]"
    if isinstance(v, dict):
        if not v:
            return "{}"
        items = [pad + json.dumps(k, ensure_ascii=False) + ": " + _dumps(val, lvl + 1) for k, val in v.items()]
        return "{\n" + ",\n".join(items) + "\n" + end + "}"
    return json.dumps(v, ensure_ascii=False)


def _parse_json_list(s):
    if s:
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return v
        except json.JSONDecodeError:
            pass
    return []


class Ideogram4Autoprompter(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Ideogram4Autoprompter",
            display_name="Ideogram 4 Autoprompter",
            category="Ideogram/text",
            search_aliases=["ideogram", "autoprompter", "caption", "bbox", "llm", "qwen", "gemini"],
            is_experimental=True,
            description="""
AI autoprompter for Ideogram 4's structured JSON caption format.

Describe an idea (and/or drop a reference image) in the node UI and let a local
vision LLM (Qwen3-VL, default) or Gemini build the full caption — background,
placed elements with bboxes, descriptions, text, and color palettes. Everything
remains editable on the canvas afterward. At run time the node only outputs the
assembled caption JSON string (plus a preview).""",
            inputs=[
                io.Int.Input("width", default=1024, min=64, max=16384, step=8,
                             tooltip="Canvas aspect width (also the pixel grid the bbox is measured in)."),
                io.Int.Input("height", default=1024, min=64, max=16384, step=8,
                             tooltip="Canvas aspect height (also the pixel grid the bbox is measured in)."),
                io.String.Input("high_level_description", multiline=True, default="",
                                tooltip="Optional one-line overview of the whole image (blank = omitted)."),
                io.String.Input("background", multiline=True, default="",
                                tooltip="Required scene background description."),
                io.DynamicCombo.Input("style", options=[
                    io.DynamicCombo.Option("none", []),
                    io.DynamicCombo.Option("photo", [
                        io.String.Input("photo", default=""),
                    ]),
                    io.DynamicCombo.Option("art_style", [
                        io.String.Input("art_style", default=""),
                    ]),
                ]),
                io.String.Input("aesthetics", default="", tooltip="Style descriptor (blank = omitted)."),
                io.String.Input("lighting", default="", tooltip="Style descriptor (blank = omitted)."),
                io.String.Input("medium", default="", tooltip="Style descriptor (blank = omitted)."),
                io.String.Input("style_palette_data", default="", socketless=True, advanced=True,
                                tooltip="Serialized style color palette from the editor (managed by the node UI)."),
                io.String.Input("elements_data", default="", socketless=True, advanced=True,
                                tooltip="Serialized regions from the editor (managed by the node UI)."),
                io.String.Input("ai_state", default="", socketless=True, advanced=True,
                                tooltip="Serialized autoprompter UI state (managed by the node UI; never the API key)."),
            ],
            outputs=[
                io.String.Output(display_name="prompt"),
                io.Image.Output(display_name="preview"),
            ],
        )

    @classmethod
    def execute(cls, width, height, background, style,
                high_level_description="", aesthetics="", lighting="", medium="",
                style_palette_data="", elements_data="", ai_state="") -> io.NodeOutput:
        boxes = _parse_json_list(elements_data)

        caption = {}
        if high_level_description.strip():
            caption["high_level_description"] = high_level_description

        kind = style["style"]                               # "none" | "photo" | "art_style"
        if kind != "none":
            sd = {"aesthetics": aesthetics, "lighting": lighting}
            if kind == "photo":
                sd["photo"] = style.get("photo", "")
                sd["medium"] = medium
            else:
                sd["medium"] = medium
                sd["art_style"] = style.get("art_style", "")
            palette = _palette(_parse_json_list(style_palette_data))
            if palette:
                sd["color_palette"] = palette
            caption["style_description"] = sd

        elements = []
        for box in boxes:
            if not isinstance(box, dict):
                continue
            etype = "text" if box.get("type") == "text" else "obj"
            elem = {"type": etype}                          # key order matters
            if not box.get("nobbox"):
                elem["bbox"] = _norm_bbox(box)
            if etype == "text":
                elem["text"] = box.get("text", "")
            elem["desc"] = box.get("desc", "")
            palette = _palette(box.get("palette", []))
            if palette:
                elem["color_palette"] = palette[:5]
            elements.append(elem)

        caption["compositional_deconstruction"] = {
            "background": background,
            "elements": elements,
        }
        preview = _render_preview(boxes, width, height)
        return io.NodeOutput(_dumps(caption), preview)
