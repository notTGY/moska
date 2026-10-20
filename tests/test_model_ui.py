import asyncio
import unittest
from unittest.mock import patch

from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from adapters.providers.openai import Account, OpenAIConnection
from adapters.providers.openai_model import ChatGPTModel
from tui.app import ChatUI
from orchestrator.demo import DemoCoordinator


class FakeConnection(OpenAIConnection):
    model = None
    effort = None

    def current_account(self):
        return Account('test-client', 'test-subject', 'test@example.com', True)

    def selected_model(self):
        return self.model

    def selected_reasoning_effort(self):
        return self.effort

    async def list_models(self):
        return [{
            'slug': slug, 'display_name': name,
            'supported_reasoning_levels': [{'effort': 'low'}, {'effort': 'high'}],
            'default_reasoning_level': 'low',
        } for slug, name in [('first', 'First model'), ('second', 'Second model')]]

    async def select_model(self, slug, reasoning_effort=None):
        self.model = slug
        self.effort = reasoning_effort


class ModelPickerTests(unittest.IsolatedAsyncioTestCase):
    async def test_picker_selects_and_activates_model(self):
        connection = FakeConnection()
        with patch('adapters.providers.registry.OpenAIConnection', return_value=connection), create_pipe_input() as pipe:
            ui = ChatUI(DemoCoordinator(), agent_factory=lambda model: model, input=pipe, output=DummyOutput())
            task = asyncio.create_task(ui.app.run_async())
            try:
                await asyncio.sleep(.05)
                pipe.send_text('/model\r')
                await asyncio.sleep(.1)
                self.assertIsNotNone(ui.picker)
                screen = ui.app.renderer._last_screen
                text = '\n'.join(''.join(screen.data_buffer[y][x].char for x in range(80)) for y in range(screen.height))
                self.assertIn('Orchestrator model', text)
                self.assertIn('First model', text)
                pipe.send_bytes(b'\x1b[B')
                pipe.send_text('\r')
                await asyncio.sleep(.1)
                self.assertIsNotNone(ui.picker)
                pipe.send_bytes(b'\x1b[B')
                pipe.send_text('\r')
                await asyncio.sleep(.1)
                self.assertIsNone(ui.picker)
                self.assertEqual(connection.model, 'second')
                self.assertEqual(ui.model, 'second')
                self.assertIsInstance(ui.agent, ChatGPTModel)
                self.assertEqual(ui.agent.model_name, 'second')
                self.assertEqual(ui.agent.reasoning_effort, 'high')
                self.assertEqual(connection.effort, 'high')
                self.assertTrue(ui.app.layout.has_focus(ui.composer))
                self.assertIn('second', ui.activity)
                pipe.send_text('/exit\r')
                await asyncio.wait_for(task, 2)
            finally:
                if not task.done():
                    ui.app.exit()
                    await task

    async def test_escape_does_not_change_selection(self):
        connection = FakeConnection()
        connection.model = 'first'
        with patch('adapters.providers.registry.OpenAIConnection', return_value=connection), create_pipe_input() as pipe:
            ui = ChatUI(DemoCoordinator(), agent_factory=lambda model: model, input=pipe, output=DummyOutput())
            task = asyncio.create_task(ui.app.run_async())
            try:
                await asyncio.sleep(.05)
                pipe.send_text('/model\r')
                await asyncio.sleep(.1)
                pipe.send_bytes(b'\x1b[B\x1b')
                await asyncio.sleep(.6)
                self.assertIsNone(ui.picker)
                self.assertEqual(connection.model, 'first')
                self.assertEqual(ui.model, 'first')
                pipe.send_text('/exit\r')
                await asyncio.wait_for(task, 2)
            finally:
                if not task.done():
                    ui.app.exit()
                    await task

    async def test_cancel_effort_keeps_previous_model_and_effort(self):
        connection = FakeConnection()
        connection.model, connection.effort = 'first', 'high'
        with patch('adapters.providers.registry.OpenAIConnection', return_value=connection), create_pipe_input() as pipe:
            ui = ChatUI(DemoCoordinator(), agent_factory=lambda model: model, input=pipe, output=DummyOutput())
            task = asyncio.create_task(ui.app.run_async())
            try:
                await asyncio.sleep(.05)
                pipe.send_text('/model\r')
                await asyncio.sleep(.1)
                pipe.send_bytes(b'\x1b[B\r')
                await asyncio.sleep(.1)
                self.assertIsNotNone(ui.picker)
                self.assertEqual(connection.model, 'first')
                pipe.send_bytes(b'\x1b')
                await asyncio.sleep(.6)
                self.assertIsNone(ui.picker)
                self.assertEqual((connection.model, connection.effort), ('first', 'high'))
                self.assertEqual(ui.agent.reasoning_effort, 'high')
                pipe.send_text('/exit\r')
                await asyncio.wait_for(task, 2)
            finally:
                if not task.done():
                    ui.app.exit()
                    await task


if __name__ == '__main__':
    unittest.main()
