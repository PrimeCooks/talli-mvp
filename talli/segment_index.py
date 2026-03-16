"""
Segment Index — defines which transformer layers are active for each task type.
For MVP we use equal (alternating) distribution across 32 layers.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default segment map for a 32-layer model (e.g. Llama 3 8B)
# ---------------------------------------------------------------------------

DEFAULT_LAYERS = 32

def _build_default_segments(num_layers: int = DEFAULT_LAYERS) -> dict[str, dict[str, list[int]]]:
    """Build an equal-alternating segment map across task types.

    Each task type gets ~half the attention and half the FFN layers,
    distributed in an interleaved pattern.
    """
    tasks = ["code", "creative", "reasoning", "chat", "multilingual"]
    all_layers = list(range(num_layers))

    segments: dict[str, dict[str, list[int]]] = {}
    for i, task in enumerate(tasks):
        # Each task gets every-Nth layer (interleaved distribution)
        attn_layers = [l for l in all_layers if l % 5 == i]
        ffn_layers = [l for l in all_layers if l % 5 == (i + 2) % 5]
        segments[task] = {
            "attention_layers": attn_layers,
            "ffn_layers": ffn_layers,
        }

    return segments


# ---------------------------------------------------------------------------
# SegmentIndex class
# ---------------------------------------------------------------------------


class SegmentIndex:
    """Load and query a segment configuration for a model."""

    def __init__(self, config_path: str | Path | None = None,
                 model_name: str = "llama3-8b",
                 num_layers: int = DEFAULT_LAYERS):
        self.model_name = model_name
        self.num_layers = num_layers
        self._config: dict[str, Any] = {}

        if config_path and Path(config_path).exists():
            self._load_from_file(config_path)
        else:
            self._build_default()

    def _load_from_file(self, path: str | Path):
        path = Path(path)
        logger.info(f"Loading segment config from {path}")
        with open(path) as f:
            self._config = json.load(f)
        self.model_name = self._config.get("model", self.model_name)
        self.num_layers = self._config.get("num_layers", self.num_layers)

    def _build_default(self):
        logger.info(f"Building default segment config for {self.model_name} ({self.num_layers} layers)")
        self._config = {
            "model": self.model_name,
            "num_layers": self.num_layers,
            "segments": _build_default_segments(self.num_layers),
        }

    # -- queries --

    @property
    def segments(self) -> dict[str, dict[str, list[int]]]:
        return self._config["segments"]

    def get_layers(self, task_type: str) -> list[int]:
        """Return *all* layer indices (attention + ffn) for a task type."""
        seg = self.segments.get(task_type)
        if seg is None:
            logger.warning(f"Unknown task type '{task_type}', returning all layers")
            return list(range(self.num_layers))
        return sorted(set(seg["attention_layers"] + seg["ffn_layers"]))

    def get_attention_layers(self, task_type: str) -> list[int]:
        seg = self.segments.get(task_type, {})
        return sorted(seg.get("attention_layers", []))

    def get_ffn_layers(self, task_type: str) -> list[int]:
        seg = self.segments.get(task_type, {})
        return sorted(seg.get("ffn_layers", []))

    def get_inactive_layers(self, task_type: str) -> list[int]:
        active = set(self.get_layers(task_type))
        return sorted(set(range(self.num_layers)) - active)

    @property
    def task_types(self) -> list[str]:
        return list(self.segments.keys())

    def layer_count(self, task_type: str) -> int:
        return len(self.get_layers(task_type))

    # -- I/O --

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._config, f, indent=2)
        logger.info(f"Saved segment config to {path}")

    def summary(self) -> str:
        lines = [f"SegmentIndex: {self.model_name} ({self.num_layers} layers)"]
        for task in self.task_types:
            attn = len(self.get_attention_layers(task))
            ffn = len(self.get_ffn_layers(task))
            lines.append(f"  {task:14s} → attn={attn} ffn={ffn}  total={attn+ffn}")
        return "\n".join(lines)
