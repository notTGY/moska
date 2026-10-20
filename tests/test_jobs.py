import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tui.jobs import WorkerCommands


class WorkerCommandsTests(unittest.TestCase):
    def setUp(self):
        self.commands = WorkerCommands()
        self.commands.add = Mock()

    def processes(self, statuses):
        return [
            {'agent_id': status + '-agent', 'agent_type': 'codex', 'workspace': '/workspace',
             'status': status, 'prompt': status + ' prompt', 'started_at': 100,
             'finished_at': None if status in ('pending', 'running') else 110}
            for status in statuses
        ]

    def test_ps_shows_only_pending_and_running_agents(self):
        rows = self.processes(('success', 'pending', 'error', 'running', 'interrupted', 'idle'))
        self.commands.worker_runtime = SimpleNamespace(processes=lambda: rows, running_count=2)

        with patch('tui.jobs.time.time', return_value=120):
            self.commands.show_processes()

        self.commands.add.assert_called_once_with('system', '\n'.join([
            '2 agents running · /ps', '', 'codex pending-agent · starting · 20s',
            'Workspace: /workspace', 'Prompt: pending prompt', '',
            'codex running-agent · running · 20s', 'Workspace: /workspace',
            'Prompt: running prompt',
        ]))

    def test_ps_reports_no_running_agents_when_only_inactive_agents_exist(self):
        rows = self.processes(('success', 'error', 'interrupted', 'idle'))
        self.commands.worker_runtime = SimpleNamespace(processes=lambda: rows, running_count=0)

        self.commands.show_processes()

        self.commands.add.assert_called_once_with('system', 'No local agents are running in this workspace.')

    def test_ps_reports_no_running_agents_without_workers(self):
        for runtime in (None, SimpleNamespace(processes=lambda: [], running_count=0)):
            with self.subTest(runtime=runtime):
                self.commands.worker_runtime = runtime
                self.commands.add.reset_mock()

                self.commands.show_processes()

                self.commands.add.assert_called_once_with('system', 'No local agents are running in this workspace.')
