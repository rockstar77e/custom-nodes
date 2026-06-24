# Ideogram Guidance

Advanced guidance for **dual-network** diffusion models (e.g. Ideogram 4, which uses a
separate conditional and unconditional UNet). One node:

**Ideogram DualModelGuider (channelwise + APG)** — category `Ideogram/guidance`,
outputs a standard `GUIDER` (drop-in for ComfyUI's guider socket on
`SamplerCustomAdvanced`).

It combines three techniques to let you push guidance higher for a punchier
photoreal look without color burn / oversaturation:

- **Channelwise CFG normalization** — per-channel std match to the conditional
  prediction (anti color-burn).
- **APG (Adaptive Projected Guidance / orthogonal CFG)** — attenuates the part of
  the guidance vector parallel to the conditional prediction (the saturation
  driver), keeping the orthogonal part.
- **CFG momentum** — running average of the guidance direction across steps.

## Install

Copy the whole `comfyui-ideogram-guidance` folder into:

```
ComfyUI/custom_nodes/comfyui-ideogram-guidance/
```

Then restart ComfyUI. No extra pip dependencies (uses torch + comfy internals only).
Verify on startup that the console lists the node and there are no import errors.

## Wiring

```
cond model   → ModelSamplingAuraFlow(shift=5) → model_cond
uncond model → ModelSamplingAuraFlow(shift=5) → model_uncond   ← SAME shift on BOTH
prompt CLIPTextEncode            → positive
CLIPTextEncode → ConditioningZeroOut → negative
Ideogram DualModelGuider         → SamplerCustomAdvanced (guider input)
```

> **Important:** both model inputs must go through *identical* ModelSampling
> patches. Each network turns the sampler's sigma into its denoised prediction
> using its own model-sampling; if they differ the two predictions are in
> mismatched spaces and the combine is incorrect.

`cfg` is a node input, so you can remove a separate `CFGOverride`.

## Parameters

| Param | Start | Meaning |
|-------|-------|---------|
| `cfg` | 4.5 | Guidance scale. Push above vanilla; rescale/APG absorb the saturation. |
| `channelwise_strength` | 0.7 | Per-channel std match. 0=off, 1=full. Anti color-burn. |
| `apg_eta` | 0.0 | Parallel retention. 1.0=plain CFG, 0.0=fully orthogonal (max anti-sat). |
| `apg_norm_threshold` | 0.0 | Optional guidance-norm clamp. 0=off; try 6–12 if highlights clip. |
| `momentum` | 0.0 | Running average across steps. 0=off; APG paper uses ≈ −0.5. |

**Identity check:** `apg_eta=1.0, channelwise_strength=0, momentum=0,
apg_norm_threshold=0` reproduces vanilla dual-model CFG at the same `cfg`. If a
fixed-seed A/B matches your stock guider with those settings, the plumbing is correct.

## Licensing / provenance

`ideogram_dual_guider.py` is an independent implementation written from published
papers (Lin et al. arXiv:2305.08891 for CFG rescale; Sadat et al. arXiv:2410.02416
for APG + momentum).
