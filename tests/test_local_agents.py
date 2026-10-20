import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prompt_toolkit.data_structures import Point
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.output import DummyOutput

from adapters.agents.registry import AGENT_SPECS, AgentSettings, AgentSpec, DetectedAgent, scan_agents
from adapters.providers.openai import OpenAIConnection
from main import ChatUI
from tui.agents import AgentSelector


class Disconnected(OpenAIConnection):
    def current_account(self):
        return None


class AgentDetectionTests(unittest.TestCase):
    def test_path_scan_and_registry_extension_do_not_launch_agents(self):
        second = AgentSpec('other', 'Other', 'other-agent', 'Install Other first.')
        with patch('adapters.agents.registry.shutil.which', side_effect=['/bin/codex', None]) as which:
            found = scan_agents((*AGENT_SPECS, second))
        self.assertEqual([entry.available for entry in found], [True, False])
        self.assertEqual([call.args[0] for call in which.call_args_list], ['codex', 'other-agent'])
        self.assertIn('codex login', found[0].hint)
        self.assertEqual(found[1].hint, second.setup_hint)

    def test_unavailable_entry_cannot_be_toggled_and_explains_setup_on_hover(self):
        entries = [DetectedAgent(AGENT_SPECS[0], '/bin/codex'), DetectedAgent(
            AgentSpec('other', 'Other', 'other-agent', 'Install Other and login first.'), None,
        )]
        selector = AgentSelector(entries, {'codex', 'other'}, lambda value: None, lambda: None)
        self.assertEqual(selector.selected, {'codex'})
        move = MouseEvent(Point(0, 1), MouseEventType.MOUSE_MOVE, MouseButton.LEFT, frozenset())
        selector.mouse(1, move)
        self.assertEqual(selector.hint, entries[1].spec.setup_hint)
        selector.toggle(1)
        self.assertEqual(selector.selected, {'codex'})
        self.assertEqual(selector.hint_index, selector.focused)
        self.assertTrue(any('disabled' in fragment[0] for fragment in selector.fragments()))


class AgentSelectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_agents_command_scan_enable_save_and_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = AgentSettings(Path(tmp) / 'agents.json')
            entries = [DetectedAgent(AGENT_SPECS[0], '/bin/codex')]
            with patch('adapters.providers.registry.OpenAIConnection', return_value=Disconnected()), patch('main.AgentSettings', return_value=settings), patch('main.scan_agents', return_value=entries) as scan, create_pipe_input() as pipe:
                ui = ChatUI(None, input=pipe, output=DummyOutput())
                task = asyncio.create_task(ui.app.run_async())
                try:
                    await asyncio.sleep(.05)
                    pipe.send_text('/agents\r')
                    await asyncio.sleep(.1)
                    self.assertIsInstance(ui.picker, AgentSelector)
                    screen = ui.app.renderer._last_screen
                    text = '\n'.join(''.join(screen.data_buffer[y][x].char for x in range(80)) for y in range(screen.height))
                    self.assertIn('On-device agents', text)
                    self.assertIn('Codex · installed', text)
                    self.assertIn('codex login', text)
                    pipe.send_text(' \r')
                    await asyncio.sleep(.1)
                    self.assertIsNone(ui.picker)
                    self.assertEqual(settings.enabled(), {'codex'})
                    pipe.send_text('/agents\r')
                    await asyncio.sleep(.1)
                    self.assertEqual(ui.picker.selected, {'codex'})
                    pipe.send_text(' ')
                    pipe.send_bytes(b'\x1b')
                    await asyncio.sleep(.6)
                    self.assertEqual(settings.enabled(), {'codex'})
                    self.assertEqual(scan.call_count, 2)
                    pipe.send_text('/exit\r')
                    await asyncio.wait_for(task, 2)
                finally:
                    if not task.done():
                        ui.app.exit()
                        await task

    async def test_missing_codex_is_disabled_and_keyboard_shows_install_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = AgentSettings(Path(tmp) / 'agents.json')
            entries = [DetectedAgent(AGENT_SPECS[0], None)]
            with patch('adapters.providers.registry.OpenAIConnection', return_value=Disconnected()), patch('main.AgentSettings', return_value=settings), patch('main.scan_agents', return_value=entries), create_pipe_input() as pipe:
                ui = ChatUI(None, input=pipe, output=DummyOutput())
                task = asyncio.create_task(ui.app.run_async())
                try:
                    await asyncio.sleep(.05)
                    pipe.send_text('/agents\r')
                    await asyncio.sleep(.1)
                    self.assertIn('Install Codex', ui.picker.hint)
                    pipe.send_text(' ')
                    await asyncio.sleep(.05)
                    self.assertEqual(ui.picker.selected, set())
                    pipe.send_bytes(b'\x1b')
                    await asyncio.sleep(.6)
                    self.assertIsNone(ui.picker)
                    self.assertFalse(settings.path.exists())
                    pipe.send_text('/exit\r')
                    await asyncio.wait_for(task, 2)
                finally:
                    if not task.done():
                        ui.app.exit()
                        await task


if __name__ == '__main__':
    unittest.main()
