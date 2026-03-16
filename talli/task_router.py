"""
Task Router — classifies queries into task types for segment selection.
Supports keyword matching (fast) and optional embedding similarity (accurate).
"""

import logging
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASK_TYPES = ["code", "creative", "reasoning", "chat", "multilingual"]

TASK_KEYWORDS: dict[str, list[str]] = {
    "code": [
        "function", "code", "python", "javascript", "java", "typescript",
        "debug", "implement", "class", "method", "api", "script",
        "programming", "compiler", "syntax", "variable", "loop", "array",
        "object", "import", "return", "def ", "async", "callback",
        "git", "linux", "sql", "html", "css", "rust", "golang", "c++",
        "bug", "error", "exception", "stack trace", "refactor",
    ],
    "creative": [
        "write", "poem", "story", "creative", "imagine", "novel",
        "character", "plot", "fiction", "narrative", "essay", "song",
        "screenplay", "dialogue", "rhyme", "metaphor", "prose",
        "creative writing", "short story", "tale",
    ],
    "reasoning": [
        "why", "how", "calculate", "prove", "explain", "logic",
        "reason", "think", "solve", "equation", "proof", "derive",
        "analyze", "compare", "evaluate", "deduce", "hypothesis",
        "theorem", "algorithm", "strategy", "step by step",
        "probability", "statistics",
    ],
    "chat": [
        "hello", "hi", "hey", "thanks", "ok", "sure", "how are you",
        "good morning", "good night", "bye", "cool", "nice", "great",
        "what's up", "how's it going", "awesome", "great",
    ],
    "multilingual": [
        "translate", "french", "spanish", "chinese", "japanese", "german",
        "language", "translate to", "in spanish", "in french", "in german",
        "hindi", "arabic", "korean", "italian", "portuguese",
    ],
}

# Context window — last N queries used for prefetch decisions
CONTEXT_WINDOW = 3

# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class TaskRouter:
    """Classify a user query into one of the supported task types."""

    def __init__(self, use_embeddings: bool = False):
        self.use_embeddings = use_embeddings
        self._embedding_model = None
        self._task_embeddings: dict[str, list[float]] = {}

        # Conversation history for context-aware classification
        self._history: list[str] = []

        if use_embeddings:
            self._init_embeddings()

    # -- embedding backend (optional, heavier) --

    def _init_embeddings(self):
        """Lazy-load sentence-transformers for semantic classification."""
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            # Pre-encode task descriptions
            task_descriptions = {
                "code": "Write code, debug, programming, implement a function or class",
                "creative": "Creative writing, poetry, storytelling, imagination",
                "reasoning": "Math, logic, analysis, step-by-step reasoning, prove",
                "chat": "Casual conversation, greetings, small talk",
                "multilingual": "Translation, foreign language, multilingual text",
            }
            for task, desc in task_descriptions.items():
                self._task_embeddings[task] = self._embedding_model.encode(desc).tolist()
            logger.info("Embedding-based task router initialized")
        except Exception as e:
            logger.warning(f"Falling back to keyword router: {e}")
            self.use_embeddings = False

    # -- public API --

    def classify(self, query: str) -> str:
        """Return the task type for *query*."""
        if self.use_embeddings and self._embedding_model:
            result = self._classify_embeddings(query)
        else:
            result = self._classify_keywords(query)

        # Update history (keep last N)
        self._history.append(result)
        if len(self._history) > CONTEXT_WINDOW:
            self._history.pop(0)

        logger.debug(f"TaskRouter: '{query[:60]}…' → {result}")
        return result

    @property
    def history(self) -> list[str]:
        return list(self._history)

    def prefetch_task(self) -> Optional[str]:
        """Return the most likely *next* task based on history (for pre-loading)."""
        if not self._history:
            return None
        counts = Counter(self._history)
        most_common = counts.most_common(1)[0]
        if most_common[1] >= 2:  # at least 2 of last 3 were same task
            return most_common[0]
        return None

    def reset_history(self):
        self._history.clear()

    # -- internals --

    def _classify_keywords(self, query: str) -> str:
        """Keyword / substring overlap scoring."""
        q_lower = query.lower()
        scores: dict[str, int] = {task: 0 for task in TASK_KEYWORDS}

        for task, keywords in TASK_KEYWORDS.items():
            for kw in keywords:
                if kw in q_lower:
                    scores[task] += 1

        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return "chat"  # default fallback
        return best

    def _classify_embeddings(self, query: str) -> str:
        """Semantic similarity via sentence-transformers."""
        q_emb = self._embedding_model.encode(query).tolist()

        import math

        def cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            return dot / (na * nb) if na and nb else 0.0

        best_task = "chat"
        best_score = -1.0
        for task, t_emb in self._task_embeddings.items():
            score = cosine(q_emb, t_emb)
            if score > best_score:
                best_score = score
                best_task = task
        return best_task


# Convenience function (mirrors spec)
_router_instance: Optional[TaskRouter] = None


def classify_task(query: str) -> str:
    """Module-level convenience — uses a singleton keyword router."""
    global _router_instance
    if _router_instance is None:
        _router_instance = TaskRouter(use_embeddings=False)
    return _router_instance.classify(query)
