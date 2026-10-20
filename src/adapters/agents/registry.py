"""Discover local agent executables and persist enabled agent types."""

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path


MIN_CODEX_VERSION = '0.163.0'


def codex_runtime(*args, **kwargs):
    from .bridge import CodexWorkers
    return CodexWorkers(*args, **kwargs)


def version_key(version):
    match = re.fullmatch(r'(\d+)\.(\d+)\.(\d+)(-[\w.-]+)?(?:\+[\w.-]+)?', version)
    if not match:
        raise ValueError('Invalid agent version')
    return (*map(int, match.group(1, 2, 3)), not bool(match[4]))


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    name: str
    executable: str
    setup_hint: str
    installed_hint: str = ''
    minimum_version: str | None = None
    runtime_factory: Callable | None = None


AGENT_SPECS = (
    AgentSpec(
        agent_id='codex', name='Codex', executable='codex',
        setup_hint=f'Install Codex {MIN_CODEX_VERSION} or newer and run codex login before using it with Moska.',
        installed_hint="Run codex login first if you haven't already.",
        minimum_version=MIN_CODEX_VERSION,
        runtime_factory=codex_runtime,
    ),
)


@dataclass(frozen=True)
class DetectedAgent:
    spec: AgentSpec
    path: str | None
    version: str | None = None

    @property
    def available(self):
        if self.path is None:
            return False
        if self.spec.minimum_version is None:
            return True
        try:
            return self.version is not None and version_key(self.version) >= version_key(self.spec.minimum_version)
        except ValueError:
            return False

    @property
    def hint(self):
        if self.path is None:
            return self.spec.setup_hint
        if not self.available:
            return (f'{self.spec.name} {self.version or "version unknown"}; requires '
                    f'{self.spec.minimum_version} or newer. Update Codex, then run codex login.')
        version = f' ({self.version})' if self.version else ''
        return f'Installed{version} at {self.path}. {self.spec.installed_hint}'.strip()


def scan_agents(specs=AGENT_SPECS):
    """Inspect PATH and bounded version probes; never start an agent or read credentials."""
    found = []
    for spec in specs:
        path = shutil.which(spec.executable)
        version = None
        if path and spec.minimum_version:
            try:
                result = subprocess.run([path, '--version'], capture_output=True, text=True,
                                        timeout=3, check=True)
                match = re.search(r'\b(\d+\.\d+\.\d+(?:-[\w.-]+)?(?:\+[\w.-]+)?)\b', result.stdout)
                version = match[1] if match else None
            except (OSError, subprocess.SubprocessError):
                pass
        found.append(DetectedAgent(spec, path, version))
    return found


def require_codex():
    entry = scan_agents((AGENT_SPECS[0],))[0]
    if not entry.available:
        from .codex import CodexError
        raise CodexError(entry.hint)
    return entry.path


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
