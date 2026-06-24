# ideogram_dual_guider.py
# Ideogram DualModelGuider — advanced guidance for dual-network models (e.g. Ideogram 4).
#
# Combines three well-documented, clean-room guidance techniques on top of a
# two-model CFG combine:
#
#   1. Channelwise CFG normalization (per-channel std match) — anti color-burn.
#         Source: Lin et al., "Common Diffusion Noise Schedules and Sample Steps
#         are Flawed" (arXiv:2305.08891), the "guidance rescale" / phi technique.
#   2. APG / Adaptive Projected Guidance (orthogonal CFG).
#         Source: Sadat et al., "Eliminating Oversaturation and Artifacts of High
#         Guidance Scales in Diffusion Models" (arXiv:2410.02416). The reference
#         implementation is permissively licensed; reimplemented here from the paper.
#   3. CFG momentum — running average of the guidance vector across steps.
#         Also from the APG paper (momentum buffer).
#
# NOTE ON LICENSING: this file is an independent implementation written from the
# published papers above. It contains NO code derived from RES4LYF (GPL-3.0).
# It is safe to ship under your own proprietary / MIT terms.
#
# WIRING REQUIREMENT: feed BOTH model inputs through *identical* ModelSampling
# patches (e.g. ModelSamplingAuraFlow shift=5 on each). Each network converts the
# sampler's sigma to a denoised x0 prediction using its OWN model_sampling; if the
# two differ, the two predictions live in different spaces and the combine is wrong.

import torch
import comfy.samplers
import comfy.model_management
from comfy.samplers import calc_cond_batch, CFGGuider


def _reduce_dims(t):
    """All dims except batch (dim 0). Used for vector norms / projection."""
    return tuple(range(1, t.dim()))


def _spatial_dims(t):
    """All dims except batch (0) and channel (1). Used for per-channel std."""
    if t.dim() <= 2:
        return tuple(range(1, t.dim()))
    return tuple(range(2, t.dim()))


def _project(diff, ref):
    """Decompose `diff` into components parallel and orthogonal to `ref`.

    Returns (parallel, orthogonal). Done per-sample over all non-batch dims,
    matching the APG paper's projection.
    """
    dims = _reduce_dims(diff)
    ref_norm = ref.norm(p=2, dim=dims, keepdim=True).clamp_min(1e-12)
    ref_unit = ref / ref_norm
    parallel = (diff * ref_unit).sum(dim=dims, keepdim=True) * ref_unit
    orthogonal = diff - parallel
    return parallel, orthogonal


class IdeogramDualModelGuider(CFGGuider):
    def __init__(self, model_patcher_cond, model_patcher_uncond):
        super().__init__(model_patcher_cond)          # sets self.model_patcher (cond)
        self.uncond_model_patcher = model_patcher_uncond
        self.inner_model_uncond = None

        # tuning params (overwritten by the node)
        self.cfg = 3.5
        self.channelwise_phi = 0.7        # 0 = off, 1 = full per-channel std match
        self.apg_eta = 0.0                # 1.0 = standard CFG, 0.0 = fully orthogonal
        self.apg_norm_threshold = 0.0     # 0 = off; clamps guidance-vector norm
        self.momentum = 0.0               # 0 = off; APG paper uses ~ -0.5
        self._momentum_buffer = None

    # ---- lifecycle ---------------------------------------------------------

    def outer_sample(self, *args, **kwargs):
        # fresh momentum state for each generation; release the second model after
        self._momentum_buffer = None
        try:
            return super().outer_sample(*args, **kwargs)
        finally:
            # mirror the primary model's cleanup so current_patcher doesn't leak
            self.uncond_model_patcher.cleanup()
            self.inner_model_uncond = None

    # ---- core --------------------------------------------------------------

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        # Lazily co-resident the uncond model with the already-loaded cond model.
        # Passing both patchers in one call keeps neither from evicting the other.
        if self.inner_model_uncond is None:
            comfy.model_management.load_models_gpu(
                [self.model_patcher, self.uncond_model_patcher] + list(self.loaded_models)
            )
            # pre_run() sets model.current_patcher, which calc_cond_batch requires
            # (model.current_patcher.prepare_state). The stock sampler does this for
            # the primary model only, so we must do it for the second network too.
            self.uncond_model_patcher.pre_run()
            self.inner_model_uncond = self.uncond_model_patcher.model

        positive = self.conds.get("positive", None)
        negative = self.conds.get("negative", None)

        # Denoised (x0-space) predictions: positive from the cond network,
        # negative from the dedicated unconditional network.
        cond_pred = calc_cond_batch(self.inner_model, [positive], x, timestep, model_options)[0]
        uncond_pred = calc_cond_batch(self.inner_model_uncond, [negative], x, timestep, model_options)[0]

        return self._combine(cond_pred, uncond_pred, x, timestep, model_options)

    def _combine(self, cond, uncond, x, timestep, model_options):
        out_dtype = cond.dtype
        cond = cond.float()
        uncond = uncond.float()
        cfg = float(self.cfg)

        diff = cond - uncond

        # 1) CFG momentum: running average of the guidance direction.
        if self.momentum != 0.0:
            if self._momentum_buffer is None or self._momentum_buffer.shape != diff.shape:
                self._momentum_buffer = torch.zeros_like(diff)
            self._momentum_buffer = diff + self.momentum * self._momentum_buffer
            diff = self._momentum_buffer

        # 2a) APG optional norm rescale: clamp the magnitude of the guidance vector.
        if self.apg_norm_threshold > 0.0:
            dims = _reduce_dims(diff)
            diff_norm = diff.norm(p=2, dim=dims, keepdim=True)
            scale = torch.minimum(
                torch.ones_like(diff_norm),
                self.apg_norm_threshold / diff_norm.clamp_min(1e-12),
            )
            diff = diff * scale

        # 2b) APG projection: attenuate the component of the guidance that is
        #     parallel to the conditional prediction (the saturation driver),
        #     keep the orthogonal component (the semantic/detail driver).
        parallel, orthogonal = _project(diff, cond)
        diff = orthogonal + self.apg_eta * parallel

        # Guided prediction (equivalent to standard CFG when eta=1 and no momentum).
        guided = cond + (cfg - 1.0) * diff

        # 3) Channelwise CFG normalization: rescale guided result so each latent
        #    channel matches the conditional prediction's per-channel std, then
        #    blend by phi. Kills color burn / highlight clipping at high cfg.
        if self.channelwise_phi > 0.0:
            sdims = _spatial_dims(guided)
            std_cond = cond.std(dim=sdims, keepdim=True)
            std_guided = guided.std(dim=sdims, keepdim=True).clamp_min(1e-12)
            rescaled = guided * (std_cond / std_guided)
            guided = self.channelwise_phi * rescaled + (1.0 - self.channelwise_phi) * guided

        # Preserve compatibility with post-CFG hooks other patchers may register.
        for fn in model_options.get("sampler_post_cfg_function", []):
            args = {
                "denoised": guided,
                "cond": self.conds.get("positive", None),
                "uncond": self.conds.get("negative", None),
                "cond_scale": cfg,
                "model": self.inner_model,
                "uncond_denoised": uncond,
                "cond_denoised": cond,
                "sigma": timestep,
                "model_options": model_options,
                "input": x,
            }
            guided = fn(args)

        return guided.to(out_dtype)


# ----------------------------------------------------------------------------
# ComfyUI node wrapper
# ----------------------------------------------------------------------------

class IdeogramDualModelGuiderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_cond": ("MODEL",),
                "model_uncond": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 30.0, "step": 0.1,
                                  "tooltip": "Guidance scale. Push higher than vanilla — the rescale/APG below absorb the saturation."}),
                "channelwise_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                                  "tooltip": "Per-channel std match to the conditional prediction. 0=off, ~0.7 is a good start. Anti color-burn."}),
                "apg_eta": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                  "tooltip": "Parallel-component retention. 1.0=plain CFG, 0.0=fully orthogonal (max anti-saturation)."}),
                "apg_norm_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 0.5,
                                  "tooltip": "Optional clamp on guidance-vector magnitude. 0=off. Try 4-15 if highlights still clip."}),
                "momentum": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                                  "tooltip": "Running average of guidance across steps. 0=off. APG paper uses ~ -0.5 to smooth oversaturation."}),
            }
        }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "Ideogram/guidance"

    def get_guider(self, model_cond, model_uncond, positive, negative,
                   cfg, channelwise_strength, apg_eta, apg_norm_threshold, momentum):
        g = IdeogramDualModelGuider(model_cond, model_uncond)
        g.set_conds(positive, negative)
        g.set_cfg(cfg)
        g.channelwise_phi = float(channelwise_strength)
        g.apg_eta = float(apg_eta)
        g.apg_norm_threshold = float(apg_norm_threshold)
        g.momentum = float(momentum)
        return (g,)


NODE_CLASS_MAPPINGS = {
    "IdeogramDualModelGuider": IdeogramDualModelGuiderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IdeogramDualModelGuider": "Ideogram DualModelGuider (channelwise + APG)",
}
