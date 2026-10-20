"""Orchestrator model connections, used by /connect and /model only."""

from .base import ProviderError
from .registry import MODEL_PROVIDERS, ProviderSpec

__all__ = ['MODEL_PROVIDERS', 'ProviderError', 'ProviderSpec']
