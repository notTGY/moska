"""Discover local agent executables and persist enabled agent types."""

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    name: str
    executable: str
    setup_hint: str
    installed_hint: str = ''


AGENT_SPECS = (
    AgentSpec(
        agent_id='codex', name='Codex', executable='codex',
        setup_hint='Install Codex and run codex login before using it with Moska.',
        installed_hint="Run codex login first if you haven't already.",
    ),
)


@dataclass(frozen=True)
class DetectedAgent:
    spec: AgentSpec
    path: str | None

    @property
    def available(self):
        return self.path is not None

    @property
    def hint(self):
        if not self.available:
            return self.spec.setup_hint
        return f'Installed at {self.path}. {self.spec.installed_hint}'.strip()


def scan_agents(specs=AGENT_SPECS):
    """Only inspect PATH; do not launch agents or read their credentials."""
    return [DetectedAgent(spec, shutil.which(spec.executable)) for spec in specs]


class AgentSettingsError(Exception):
    pass


class AgentSettings:
    def __init__(self, path: Path | None = None):
        base = Path(os.environ.get('XDG_CONFIG_HOME', str(Path.home() / '.config')))
        self.path = path or base / 'moska' / 'agents.json'

    def enabled(self) -> set[str]:
        try:
            data = json.loads(self.path.read_text())
            values = data['enabled']
            if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                raise ValueError
            return set(values)
        except FileNotFoundError:
            return set()
        except (OSError, ValueError, KeyError, TypeError):
            raise AgentSettingsError('Could not read local agent settings.') from None

    def save(self, enabled: set[str]):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd, name = tempfile.mkstemp(dir=self.path.parent, prefix='.agents-')
            try:
                with os.fdopen(fd, 'w') as file:
                    os.fchmod(file.fileno(), 0o600)
                    json.dump({'enabled': sorted(enabled)}, file)
                os.replace(name, self.path)
            finally:
                if os.path.exists(name):
                    os.unlink(name)
        except OSError:
            raise AgentSettingsError('Could not save local agent settings.') from None
