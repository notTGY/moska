import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from prompt_toolkit.data_structures import Point
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.output import DummyOutput

from orchestrator.storage import SessionStore
from tui.app import ChatUI
from tui.jobs import WorkerCommands, process_fragments
from tui.transcript import SelectableTranscript


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

        self.commands.add.assert_called_once_with('agents', '\n'.join([
            'Local agents · 2 active', '', 'codex · pending-agent',
            '  Status: starting · 20s', '  Workspace: /workspace', '  Prompt:',
            '    pending prompt', '', 'codex · running-agent',
            '  Status: running · 20s', '  Workspace: /workspace', '  Prompt:',
            '    running prompt',
        ]))

    def test_ps_styles_snapshot_without_interpreting_or_truncating_prompt(self):
        rows = self.processes(('pending', 'running'))
        prompt = '**literal** `code`\n  Status: running\n# Heading\n\n' + '界' * 200
        rows[0]['prompt'] = prompt
        rows[0]['workspace'] = '/workspace with spaces/' + 'long/' * 30
        rows[1]['prompt'] = ''
        rows[1]['started_at'] = None
        self.commands.worker_runtime = SimpleNamespace(processes=lambda: rows, running_count=2)
        with patch('tui.jobs.time.time', return_value=120):
            self.commands.show_processes()
        role, text = self.commands.add.call_args.args
        self.assertEqual(role, 'agents')
        fragments = process_fragments(text)
        self.assertEqual(''.join(item[1] for item in fragments), text)
        self.assertIn(('class:process-summary', 'Local agents · 2 active\n'), fragments)
        self.assertIn(('class:agent', 'codex · pending-agent\n'), fragments)
        self.assertIn(('class:process-pending', 'starting · 20s\n'), fragments)
        self.assertIn(('class:process-running', 'running · —\n'), fragments)
        self.assertIn(('class:process-workspace', rows[0]['workspace'] + '\n'), fragments)
        self.assertIn('    —', text)
        self.assertIn(('class:process-label', '  Prompt:\n'), fragments)
        for line in prompt.split('\n'):
            self.assertIn(('', '    ' + line + '\n'), fragments)
        # Removing only presentation indentation recovers the exact prompt.
        prompt_block = text.split('  Prompt:\n', 1)[1].split('\n\ncodex', 1)[0]
        self.assertEqual('\n'.join(line[4:] for line in prompt_block.split('\n')), prompt)

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

    def test_ps_colors_and_copyable_snapshot_survive_session_restore(self):
        rows = self.processes(('running',))
        rows[0]['prompt'] = '**literal**\n  keep indentation\nhttps://example.com/task'
        runtime = SimpleNamespace(processes=lambda: rows, running_count=1)
        connection = SimpleNamespace(current_account=lambda: None)
        with tempfile.TemporaryDirectory() as directory:
            session = SessionStore(directory, data_dir=directory).open()
            try:
                with patch('adapters.providers.registry.OpenAIConnection', return_value=connection), create_pipe_input() as pipe:
                    ui = ChatUI(None, session=session, worker_runtime=runtime,
                                input=pipe, output=DummyOutput())
                    with patch('tui.jobs.time.time', return_value=120):
                        ui.show_processes()
                    before = ui.render_transcript()
                    restored = ChatUI(None, session=session, input=pipe, output=DummyOutput())
                    after = restored.render_transcript()
                    self.assertEqual([item[:2] for item in before], [item[:2] for item in after])
                    self.assertIn(('class:process-running', 'running · 20s\n'), after)
                    self.assertIn(('', '    **literal**\n'), after)
                    self.assertTrue(any(len(item) == 3 for item in after))
                    text = ''.join(item[1] for item in after)
                    copied = []
                    control = SelectableTranscript(lambda: after, lambda: None, copied.append)
                    for kind, point in ((MouseEventType.MOUSE_DOWN, Point(0, 0)),
                                        (MouseEventType.MOUSE_UP, Point(0, len(text.splitlines())))):
                        control.mouse_handler(MouseEvent(point, kind, MouseButton.LEFT, frozenset()))
                    self.assertEqual(copied, [text])
            finally:
                session.close()
