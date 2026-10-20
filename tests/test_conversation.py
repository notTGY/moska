import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.utils import get_cwidth
from pydantic import Field

from orchestrator.runtime import OrchestratorRuntime
from tui.app import ChatUI
from tui.progress import ResponseProgress, prompt_lines


class SteerModel(BaseChatModel):
    requests: list = Field(default_factory=list)
    started: asyncio.Event = Field(default_factory=asyncio.Event)
    block_first: bool = True

    @property
    def _llm_type(self):
        return 'steering-test'

    def _generate(self, messages, **kwargs):
        raise NotImplementedError

    async def _agenerate(self, messages, **kwargs):
        self.requests.append([message.content for message in messages if isinstance(message, HumanMessage)])
        self.started.set()
        if self.block_first and len(self.requests) == 1:
            await asyncio.Event().wait()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content='steered'))])


class PromptPreviewTests(unittest.TestCase):
    def test_three_lines_preserve_newlines_and_show_truncation(self):
        progress = ResponseProgress(0, lambda: None)
        progress.requested({'id': 'run', 'name': 'run', 'args': {
            'agent_id': 'worker', 'prompt': 'first line\nsecond line\nthird line\nfourth line'}})
        text = ''.join(fragment[1] for fragment in progress.render(True, width=60))
        details = [line for line in text.splitlines() if '↳' in line]
        self.assertEqual(len(details), 3)
        self.assertIn('Prompt: first line', details[0])
        self.assertIn('second line', details[1])
        self.assertIn('third line…', details[2])
        self.assertNotIn('fourth line', text)
        self.assertEqual(progress.events[-1]['prompt'].splitlines()[-1], 'fourth line')

    def test_wide_characters_respect_terminal_width(self):
        lines = prompt_lines('界' * 40, 10)
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[-1].endswith('…'))
        self.assertTrue(all(get_cwidth(line) <= 10 for line in lines))


class SteeringPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_rapid_followups_keep_order_and_are_not_duplicated_after_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            connection = SimpleNamespace(current_account=lambda: None)
            with patch('adapters.providers.registry.OpenAIConnection', return_value=connection):
                async with OrchestratorRuntime(workspace, data_dir=workspace / 'data') as runtime:
                    model = SteerModel()
                    agent = create_agent(model, checkpointer=runtime.memory)
                    with create_pipe_input() as pipe:
                        ui = ChatUI(agent, session=runtime.session, prepare_turn=runtime.prepare_turn,
                                    input=pipe, output=DummyOutput())
                        app = asyncio.create_task(ui.app.run_async())
                        try:
                            await asyncio.sleep(.05)
                            pipe.send_text('initial\r')
                            await asyncio.wait_for(model.started.wait(), 1)
                            pipe.send_text('first followup\rsecond followup\r')
                            for _ in range(200):
                                if len(ui.progress) > 1 and ui.progress[-1].finished:
                                    break
                                await asyncio.sleep(.01)
                            self.assertEqual(model.requests[-1], ['initial', 'first followup', 'second followup'])
                            self.assertEqual(ui.progress[0].stage, 'Steered')
                            self.assertEqual(ui.progress[-1].stage, 'Responded')
                            self.assertEqual(runtime.session.meta('pending_messages'), [])
                            config = ui.config
                            pipe.send_text('/exit\r')
                            await asyncio.wait_for(app, 2)
                        finally:
                            if not app.done():
                                ui.app.exit()
                                await app
                async with OrchestratorRuntime(workspace, data_dir=workspace / 'data') as runtime:
                    agent = create_agent(SteerModel(block_first=False), checkpointer=runtime.memory)
                    state = (await agent.aget_state(config)).values
                    messages = [message for message in state['messages'] if isinstance(message, HumanMessage)]
                    self.assertEqual([message.content for message in messages], ['initial', 'first followup', 'second followup'])
                    self.assertEqual(len({message.id for message in messages}), 3)

    async def test_unsent_followup_is_recovered_from_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            async with OrchestratorRuntime(workspace, data_dir=workspace / 'data') as runtime:
                runtime.session.append('message', {'role': 'you', 'text': 'saved before graph started'})
                runtime.session.set_meta('pending_messages', [{'id': 'pending', 'content': 'saved before graph started'}])
            with patch('adapters.providers.registry.OpenAIConnection', return_value=SimpleNamespace(current_account=lambda: None)):
                async with OrchestratorRuntime(workspace, data_dir=workspace / 'data') as runtime:
                    model = SteerModel(block_first=False)
                    agent = create_agent(model, checkpointer=runtime.memory)
                    with create_pipe_input() as pipe:
                        ui = ChatUI(agent, session=runtime.session, prepare_turn=runtime.prepare_turn,
                                    input=pipe, output=DummyOutput())
                        app = asyncio.create_task(ui.app.run_async())
                        try:
                            await asyncio.sleep(.05)
                            pipe.send_text('continue\r')
                            for _ in range(200):
                                if ui.progress and ui.progress[-1].finished:
                                    break
                                await asyncio.sleep(.01)
                            self.assertEqual(model.requests[-1], ['saved before graph started', 'continue'])
                            self.assertEqual(runtime.session.meta('pending_messages'), [])
                            pipe.send_text('/exit\r')
                            await asyncio.wait_for(app, 2)
                        finally:
                            if not app.done():
                                ui.app.exit()
                                await app
