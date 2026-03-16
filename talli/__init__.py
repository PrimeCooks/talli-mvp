"""
TALLI — Task-Adaptive LLM Inference

Only load the model segments you need for each task.
Run 70B models on consumer hardware with intelligent segment routing.
"""

__version__ = "0.1.0"

from .task_router import TaskRouter, classify_task
from .segment_index import SegmentIndex

# Lazy imports (require torch)
def __getattr__(name):
    if name == "MemoryManager":
        from .memory_manager import MemoryManager
        return MemoryManager
    if name == "TALLIInference":
        from .inference_engine import TALLIInference
        return TALLIInference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "TaskRouter",
    "classify_task",
    "SegmentIndex",
    "MemoryManager",
    "TALLIInference",
]
