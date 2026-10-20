"""Local worker discovery and configuration, used by /agents only."""

from .registry import AGENT_SPECS, AgentSettings, AgentSettingsError, AgentSpec, DetectedAgent, scan_agents

__all__ = ['AGENT_SPECS', 'AgentSettings', 'AgentSettingsError', 'AgentSpec', 'DetectedAgent', 'scan_agents']
