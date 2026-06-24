"""ComfyUI-Ideogram-Autoprompter — AI autoprompter for Ideogram 4's JSON caption format."""

import logging

from .nodes.autoprompter_node import Ideogram4Autoprompter

NODE_CLASS_MAPPINGS = {
    "Ideogram4Autoprompter": Ideogram4Autoprompter,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Ideogram4Autoprompter": "Ideogram 4 Autoprompter",
}

# Register the generation HTTP routes on PromptServer (import has the side effect).
try:
    from .nodes import routes  # noqa: F401
except Exception as e:  # pragma: no cover - server may be unavailable in some contexts
    logging.warning("Ideogram Autoprompter: routes not registered (%s)", e)

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
