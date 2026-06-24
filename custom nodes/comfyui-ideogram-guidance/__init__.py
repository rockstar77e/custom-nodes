"""Ideogram Guidance — custom guidance nodes for dual-network diffusion models.

ComfyUI discovers a custom node pack by importing this package and reading
NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS from it. We re-export them
from the implementation module so everything stays in one place.
"""

from .ideogram_dual_guider import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
