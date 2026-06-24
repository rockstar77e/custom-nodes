"""Local vision-LLM backend: Huihui-Qwen3-VL-4B-Instruct-abliterated via transformers.

Loaded on demand, optionally 4-bit, and unloaded after each generation by default so
it doesn't hold VRAM during the rest of the workflow. Download progress (first run only)
is reported through a module-level status dict polled by the frontend.
"""

import gc
import re
import threading

from .caption_schema import build_user_prompt, get_system_prompt, parse_caption

DEFAULT_MODEL = "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated"

_LOCK = threading.Lock()
_STATE = {"model": None, "processor": None, "model_id": None, "four_bit": None, "attn": None}

# Polled by GET /ideogram_autoprompter/local/status
STATUS = {"stage": "idle", "detail": "", "model_id": DEFAULT_MODEL}


def _set_status(stage, detail=""):
    STATUS["stage"] = stage
    STATUS["detail"] = detail


def is_loaded():
    return _STATE["model"] is not None


def unload():
    with _LOCK:
        _STATE["model"] = None
        _STATE["processor"] = None
        _STATE["model_id"] = None
        _STATE["four_bit"] = None
        _STATE["attn"] = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    _set_status("idle")


def _pick_device_dtype(four_bit):
    import torch
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return "auto", dtype
    return "cpu", torch.float32


def _flash_available():
    try:
        import importlib.util
        return importlib.util.find_spec("flash_attn") is not None
    except Exception:
        return False


def _resolve_attn(mode, force_sdpa):
    """Map the UI's attention choice to a transformers attn_implementation string.

    Flash-Attention 2 is the fastest but only works in fp16/bf16 and must be installed;
    when it can't be used we fall back to SDPA (PyTorch's built-in fused attention, which
    is still much faster than the 'eager' path). 4-bit/8-bit models force SDPA.
    """
    if force_sdpa:
        return "sdpa"
    if mode == "eager":
        return "eager"
    if mode == "sdpa":
        return "sdpa"
    if mode == "flash_attention_2":
        return "flash_attention_2" if _flash_available() else "sdpa"
    # auto: prefer flash if present, else sdpa
    return "flash_attention_2" if _flash_available() else "sdpa"


def _load_error_message(model_id, e):
    """Turn a from_pretrained failure into a message that names the real cause.

    A download/connection failure must NOT be reported as "not a vision model" —
    that was misleading users whose model was fine but whose HF endpoint was down.
    """
    import os

    msg = str(e)
    low = msg.lower()

    network_signs = (
        "couldn't connect", "could not connect", "connection error", "max retries",
        "failed to establish", "name resolution", "timed out", "offline mode",
        "localentrynotfound", "we couldn't connect", "hf-mirror",
    )
    if any(s in low for s in network_signs):
        endpoint = os.environ.get("HF_ENDPOINT")
        hint = ""
        if endpoint and "huggingface.co" not in endpoint:
            hint = (
                " Your HF_ENDPOINT is set to %s — that mirror looks unreachable. "
                "Run `unset HF_ENDPOINT` in the terminal you launch ComfyUI from and "
                "restart, then retry." % endpoint
            )
        else:
            hint = (
                " Check your internet connection (or HF_ENDPOINT) and retry; the model "
                "downloads several GB on first run."
            )
        return (
            "Could not download '%s' — network/endpoint problem, not a model problem.%s "
            "Underlying error: %s" % (model_id, hint, msg)
        )

    arch_signs = (
        "unrecognized", "does not recognize", "not a vision", "no module named",
        "keyerror", "architecture", "imagetexttotext", "vision",
        "got an unexpected keyword", "object has no attribute",
    )
    if any(s in low for s in arch_signs):
        return (
            "Could not load '%s' as a vision model. This node needs a multimodal / "
            "vision LLM — e.g. huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated, "
            "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated, or "
            "Qwen/Qwen3-VL-4B-Instruct. Plain text LLMs (Qwen3 / Qwen3.5) are not "
            "supported. Underlying error: %s" % (model_id, msg)
        )

    return "Could not load '%s'. Underlying error: %s" % (model_id, msg)


def _load(model_id, four_bit, attn):
    """Load model+processor into _STATE. Caller holds _LOCK."""
    if (_STATE["model"] is not None and _STATE["model_id"] == model_id
            and _STATE["four_bit"] == four_bit and _STATE["attn"] == attn):
        return
    if _STATE["model"] is not None:
        _STATE["model"] = None
        _STATE["processor"] = None
        gc.collect()

    import torch
    from transformers import AutoProcessor

    _set_status("downloading", "Fetching %s (first run downloads several GB)…" % model_id)
    device_map, dtype = _pick_device_dtype(four_bit)

    quant = None
    if four_bit and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        except Exception:
            quant = None

    # Architecture-agnostic: let transformers dispatch to the right class from the model's
    # own config (Qwen3-VL, Gemma 3/3n, LLaVA, etc.). The dedicated Qwen3-VL class is only a
    # fallback for older transformers where the Auto map didn't include Qwen3-VL yet.
    try:
        from transformers import AutoModelForImageTextToText as _VLModel
    except Exception:
        from transformers import Qwen3VLForConditionalGeneration as _VLModel

    # Flash-Attn needs fp16/bf16 weights, so quantized loads force SDPA.
    attn_impl = _resolve_attn(attn, force_sdpa=(quant is not None))

    kwargs = {
        "device_map": device_map,
        "dtype": dtype,
        "trust_remote_code": True,
        "attn_implementation": attn_impl,
    }
    if quant is not None:
        kwargs["quantization_config"] = quant
        kwargs.pop("dtype", None)

    _set_status("loading", "Loading model (%s attention)…" % attn_impl)
    try:
        model = _VLModel.from_pretrained(model_id, **kwargs)
    except Exception as e:
        # Flash-Attn often installs but fails to load (kernel/CUDA mismatch); retry on SDPA
        # before giving up so a bad flash build never blocks generation.
        if attn_impl == "flash_attention_2":
            kwargs["attn_implementation"] = "sdpa"
            try:
                model = _VLModel.from_pretrained(model_id, **kwargs)
                attn_impl = "sdpa"
            except Exception as e2:
                raise RuntimeError(_load_error_message(model_id, e2)) from e2
        else:
            raise RuntimeError(_load_error_message(model_id, e)) from e
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    _STATE["model"] = model
    _STATE["processor"] = processor
    _STATE["model_id"] = model_id
    _STATE["four_bit"] = four_bit
    _STATE["attn"] = attn
    _set_status("ready")


def _apply_template(processor, messages, think):
    """Tokenize a message list. When think is False, ask the chat template to disable
    reasoning (Qwen3-style `enable_thinking`); processors that don't accept the kwarg
    are retried without it."""
    base = dict(tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt")
    if not think:
        try:
            return processor.apply_chat_template(messages, enable_thinking=False, **base)
        except Exception:
            pass
    return processor.apply_chat_template(messages, **base)


def _build_inputs(processor, model, sys_text, user_text, pil_image, think):
    """Tokenize one VLM turn, working across model families.

    Most chat templates accept a separate ``system`` role, but some popular vision
    models (e.g. Gemma 3 / 3n) reject it. When that happens we fold the system text
    into the user turn so any vision model can still be driven the same way.
    """
    user_content = []
    if pil_image is not None:
        user_content.append({"type": "image", "image": pil_image})
    user_content.append({"type": "text", "text": user_text})

    with_system = [
        {"role": "system", "content": [{"type": "text", "text": sys_text}]},
        {"role": "user", "content": user_content},
    ]
    try:
        inputs = _apply_template(processor, with_system, think)
    except Exception:
        # Fold the system prompt into the user turn for models without a system role.
        merged = [{"type": "text", "text": sys_text + "\n\n"}]
        if pil_image is not None:
            merged.append({"type": "image", "image": pil_image})
        merged.append({"type": "text", "text": user_text})
        inputs = _apply_template(processor, [{"role": "user", "content": merged}], think)

    return inputs.to(model.device)


def _make_streamer(processor, model_id):
    """A TextStreamer that prints generated tokens to the ComfyUI console live."""
    try:
        import sys
        from transformers import TextStreamer
        tok = getattr(processor, "tokenizer", None) or processor
        sys.stdout.write("\n[ideogram-autoprompter] local '%s' streaming:\n" % model_id)
        sys.stdout.flush()
        return TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    except Exception:
        return None


def generate(idea, pil_image=None, model_id=None, four_bit=False, unload_after=True,
             density="normal", attn="auto", think=False, stream_to_console=True,
             max_new_tokens=2048):
    """Run the local VLM and return a normalized caption dict."""
    import torch

    model_id = model_id or DEFAULT_MODEL
    STATUS["model_id"] = model_id
    with _LOCK:
        try:
            _load(model_id, four_bit, attn)
            model, processor = _STATE["model"], _STATE["processor"]

            _set_status("generating", "Generating caption…")
            sys_text = get_system_prompt(density)
            user_text = build_user_prompt(idea, pil_image is not None)
            inputs = _build_inputs(processor, model, sys_text, user_text, pil_image, think)

            streamer = _make_streamer(processor, model_id) if stream_to_console else None
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    streamer=streamer,
                )
            trimmed = generated[:, inputs["input_ids"].shape[1]:]
            text = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            # Drop any reasoning the model emitted before its JSON answer.
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

            caption = parse_caption(text)
            _set_status("ready")
            return caption
        finally:
            if unload_after:
                _STATE["model"] = None
                _STATE["processor"] = None
                _STATE["model_id"] = None
                _STATE["four_bit"] = None
                _STATE["attn"] = None
                gc.collect()
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                _set_status("idle")
