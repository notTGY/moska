import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from tui.app import ChatUI


class ChatScrollTests(unittest.IsolatedAsyncioTestCase):
    async def test_wheel_and_page_keys_scroll_and_preserve_history_position(self):
        connection = SimpleNamespace(current_account=lambda: None)
        with patch('adapters.providers.registry.OpenAIConnection', return_value=connection), create_pipe_input() as pipe:
            ui = ChatUI(None, input=pipe, output=DummyOutput())
            ui.add('moska', '\n'.join(f'history {i}' for i in range(100)))
            window = next(window for window in ui.app.layout.find_all_windows()
                          if window.content is ui.transcript)
            task = asyncio.create_task(ui.app.run_async())
            try:
                await asyncio.sleep(.05)
                initial_scroll = window.vertical_scroll
                self.assertGreater(initial_scroll, 0)

                # SGR wheel events over the transcript exercise the window's
                # mouse-coordinate translation as well as the control handler.
                pipe.send_bytes(b'\x1b[<64;4;5M' * 15)
                await asyncio.sleep(.05)
                self.assertLess(window.vertical_scroll, initial_scroll)
                self.assertFalse(ui.follow_output)
                self.assertTrue(ui.app.layout.has_focus(ui.composer))
                position, scroll = ui.cursor_line, window.vertical_scroll
                ui.add('moska', 'new output while reading history')
                ui.refresh_progress()
                await asyncio.sleep(.05)
                self.assertEqual(ui.cursor_line, position)
                self.assertEqual(window.vertical_scroll, scroll)

                pipe.send_bytes(b'\x1b[5~' * 20)
                await asyncio.sleep(.05)
                self.assertEqual(ui.cursor_line, 0)
                self.assertEqual(window.vertical_scroll, 0)
                pipe.send_bytes(b'\x1b[6~')
                await asyncio.sleep(.05)
                self.assertEqual(ui.cursor_line, 10)
                self.assertFalse(ui.follow_output)
                pipe.send_bytes(b'\x1b[<65;4;5M' * 50)
                await asyncio.sleep(.05)
                self.assertEqual(ui.cursor_line, ui.last_line)
                self.assertTrue(ui.follow_output)
                ui.add('moska', 'latest output')
                await asyncio.sleep(.05)
                self.assertEqual(ui.cursor_line, ui.last_line)
                self.assertGreater(window.vertical_scroll, initial_scroll)
                pipe.send_text('draft')
                await asyncio.sleep(.05)
                self.assertEqual(ui.composer.text, 'draft')
            finally:
                ui.app.exit()
                await task
