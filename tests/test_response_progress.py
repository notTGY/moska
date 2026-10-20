import asyncio
import unittest
from unittest.mock import patch

import httpx
from langgraph_sdk.client import LangGraphClient
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from adapters.openai import OpenAIConnection
from decomposer import create_decomposer_agent
from main import ChatUI, DemoCoordinator, MockWorkers


class Disconnected(OpenAIConnection):
    def current_account(self):
        return None


class WaitingAgent:
    def __init__(self):
        self.started = asyncio.Event()
        self.interrupted = asyncio.Event()

    async def astream(self, *args, **kwargs):
        self.started.set()
        try:
            await asyncio.Event().wait()
            yield {}
        finally:
            self.interrupted.set()


class ResponseProgressTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_tools_agent_reports_and_ctrl_t_toggle(self):
        workers = MockWorkers()
        async with httpx.AsyncClient(transport=httpx.MockTransport(workers.handle), base_url='http://mock.local') as client:
            with patch('decomposer.core.get_client', return_value=LangGraphClient(client)), patch('main.OpenAIConnection', return_value=Disconnected()), create_pipe_input() as pipe:
                agent = create_decomposer_agent(decomposer_model=DemoCoordinator(), agent_types=[{
                    'agent_type_id': 'mock', 'description': 'mock', 'assistant_id': 'mock', 'url': 'http://mock.local',
                }], agent_run_budget_seconds=None)
                ui = ChatUI(agent, input=pipe, output=DummyOutput())
                task = asyncio.create_task(ui.app.run_async())
                try:
                    await asyncio.sleep(.05)
                    pipe.send_text('hello\r')
                    for _ in range(100):
                        if ui.progress and ui.progress[-1].finished:
                            break
                        await asyncio.sleep(.01)
                    self.assertEqual(ui.progress[-1].stage, 'Responded')
                    self.assertEqual(len(ui.progress[-1].tools), 3)
                    self.assertEqual(len(ui.progress[-1].agents), 1)
                    compact = ''.join(f[1] for f in ui.render_transcript())
                    self.assertIn('▶ run', compact)
                    self.assertIn('◷ wait', compact)
                    self.assertIn('✓ Responded', compact)
                    self.assertNotIn('✦ new', compact)
                    self.assertNotIn('Requested new', compact)
                    pipe.send_bytes(b'\x14')
                    await asyncio.sleep(.1)
                    expanded = ''.join(f[1] for f in ui.render_transcript())
                    for expected in ('✦ new', '▶ run', '◷ wait', 'created', 'collected 1 report', ': hello', 'Prompt: hello'):
                        self.assertIn(expected, expanded)
                    self.assertTrue(any(f[0] == 'class:agent' for f in ui.render_transcript()))
                    self.assertNotRegex(expanded, r'\d+\.\d+s')
                    pipe.send_bytes(b'\x14')
                    await asyncio.sleep(.05)
                    self.assertFalse(ui.expanded_tools)
                    self.assertNotIn('✦ new', ''.join(f[1] for f in ui.render_transcript()))
                    pipe.send_text('/exit\r')
                    await asyncio.wait_for(task, 2)
                finally:
                    if not task.done():
                        ui.app.exit()
                        await task

    async def test_escape_interrupts_execution_and_keeps_app_open(self):
        with patch('main.OpenAIConnection', return_value=Disconnected()), create_pipe_input() as pipe:
            agent = WaitingAgent()
            ui = ChatUI(agent, input=pipe, output=DummyOutput())
            task = asyncio.create_task(ui.app.run_async())
            try:
                await asyncio.sleep(.05)
                pipe.send_text('hello\r')
                await asyncio.wait_for(agent.started.wait(), 1)
                pipe.send_bytes(b'\x1b')
                await asyncio.wait_for(agent.interrupted.wait(), 2)
                await asyncio.sleep(.05)
                self.assertEqual(ui.progress[-1].stage, 'Interrupted')
                self.assertFalse(task.done())
                self.assertTrue(ui.task.cancelled())
                pipe.send_text('/exit\r')
                await asyncio.wait_for(task, 2)
            finally:
                if not task.done():
                    ui.app.exit()
                    await task

    async def test_exit_while_busy_cancels_execution_and_closes_app(self):
        with patch('main.OpenAIConnection', return_value=Disconnected()), create_pipe_input() as pipe:
            agent = WaitingAgent()
            ui = ChatUI(agent, input=pipe, output=DummyOutput())
            task = asyncio.create_task(ui.app.run_async())
            try:
                await asyncio.sleep(.05)
                pipe.send_text('hello\r')
                await asyncio.wait_for(agent.started.wait(), 1)
                pipe.send_text('/exit\r')
                await asyncio.wait_for(task, 2)
                self.assertTrue(agent.interrupted.is_set())
                self.assertTrue(ui.task.cancelled())
            finally:
                if not task.done():
                    ui.app.exit()
                    await task


if __name__ == '__main__':
    unittest.main()
