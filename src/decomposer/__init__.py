"""Public Decomposer API with lazy imports for lightweight CLI consumers."""

from __future__ import annotations

from typing import Any

__all__ = [
    "create_decomposer_agent",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from .core import create_decomposer_agent

    return create_decomposer_agent
