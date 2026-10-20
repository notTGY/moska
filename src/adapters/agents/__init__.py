"""Local worker discovery, configuration and execution; separate from providers."""

from .registry import AGENT_SPECS, AgentSettings, AgentSettingsError, AgentSpec, DetectedAgent, scan_agents
from .runtime import LocalWorkers, worker_clients

__all__ = ['AGENT_SPECS', 'AgentSettings', 'AgentSettingsError', 'AgentSpec', 'DetectedAgent',
           'scan_agents', 'LocalWorkers', 'worker_clients']
