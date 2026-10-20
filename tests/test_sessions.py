import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from orchestrator.compaction import ContextCompaction
from orchestrator.demo import DemoCoordinator
from orchestrator.runtime import OrchestratorRuntime
from orchestrator.storage import SessionBusyError, SessionStore
from tui.app import ChatUI
from test_compaction import RecordingModel
from test_codex_bridge import FAKE_CODEX, ToolCycle
from test_model_ui import FakeConnection
from adapters.agents import AgentSettings
from decomposer import core


class SessionStorageTests(unittest.TestCase):
    def test_workspace_identity_lock_permissions_and_inspectable_history(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            workspace = base / 'workspace'
            workspace.mkdir()
            link = base / 'alias'
            link.symlink_to(workspace, target_is_directory=True)
            store = SessionStore(workspace, data_dir=base / 'data').open()
            alias = SessionStore(link, data_dir=base / 'data')
            self.assertEqual(store.session_id, alias.session_id)
            self.assertNotEqual(store.session_id, SessionStore(base, data_dir=base / 'data').session_id)
            try:
                with self.assertRaises(SessionBusyError):
                    alias.open()
                store.event('tool', {'message': ToolMessage(content='result', tool_call_id='call')})
                store.append('message', {'role': 'you', 'text': 'remember this'})
                if os.name == 'posix':
                    self.assertEqual(store.path.stat().st_mode & 0o777, 0o600)
                self.assertIn('remember this', json.dumps(store.export()))
            finally:
                store.close()
            alias.open()
            self.assertEqual(alias.export()['events'][0]['kind'], 'tool')
            alias.close()


class SessionResumeTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_graph_and_mock_workers_survive_reopening(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            config = {'configurable': {'thread_id': 'persistent'}}
            async with OrchestratorRuntime(workspace, data_dir=workspace / 'data') as runtime:
                agent = runtime.create_agent(DemoCoordinator())
                first = await agent.ainvoke({'messages': [HumanMessage(content='remember first')], 'agents': {}, 'agent_runs': {}}, config)
                original_agent = next(iter(first['agents']))
            async with OrchestratorRuntime(workspace, data_dir=workspace / 'data') as runtime:
                agent = runtime.create_agent(DemoCoordinator())
                await runtime.prepare_turn(agent, config)
                previous = (await agent.aget_state(config)).values
                self.assertIn(original_agent, previous['agents'])
                self.assertEqual(previous['messages'][0].text, 'remember first')
                second = await agent.ainvoke({'messages': [HumanMessage(content='continue')], 'agents': {}, 'agent_runs': {}}, config)
                self.assertEqual(len(second['agents']), 2)
                self.assertEqual(second['messages'][-1].text, 'hello')
                self.assertTrue(runtime.session.workers('mock', 'run'))
                self.assertTrue(runtime.session.tools('persistent'))

    async def test_compaction_summary_boundary_and_archive_survive_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            config = {'configurable': {'thread_id': 'compact'}}
            async with OrchestratorRuntime(workspace, data_dir=workspace / 'data') as runtime:
                model = RecordingModel()
                agent = create_agent(model, middleware=[ContextCompaction(model, trigger_tokens=500, keep_tokens=180, summary_tokens=100)], checkpointer=runtime.memory)
                await agent.ainvoke({'messages': [HumanMessage(content='old details ' * 250), AIMessage(content='done'), HumanMessage(content='next')]}, config)
                before = (await agent.aget_state(config)).values
                self.assertEqual(before['context_compactions'], 1)
            async with OrchestratorRuntime(workspace, data_dir=workspace / 'data') as runtime:
                model = RecordingModel()
                agent = create_agent(model, middleware=[ContextCompaction(model, trigger_tokens=500, keep_tokens=180, summary_tokens=100)], checkpointer=runtime.memory)
                after = (await agent.aget_state(config)).values
                for field in ('context_summary', 'context_summary_end', 'context_compactions'):
                    self.assertEqual(before[field], after[field])
                self.assertEqual(after['messages'][0].text, 'old details ' * 250)
                await agent.ainvoke({'messages': [HumanMessage(content='continue')]}, config)
                self.assertIn('worker-1', model.requests[-1][0].text)

    async def test_interrupted_tool_ack_and_native_worker_context_survive_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            settings = AgentSettings(workspace / 'settings.json')
            settings.save({'codex'})
            config = {'configurable': {'thread_id': 'interrupted'}}
            with patch('adapters.agents.bridge.require_codex', return_value=FAKE_CODEX):
                async with OrchestratorRuntime(workspace, settings=settings, data_dir=workspace / 'data') as runtime:
                    agent = runtime.create_agent(ToolCycle(worker_prompt='slow-start'))
                    task = asyncio.create_task(agent.ainvoke({'messages': [HumanMessage(content='inspect')], 'agents': {}, 'agent_runs': {}}, config))
                    workers = runtime.workers.runtimes['codex']
                    for _ in range(200):
                        if workers.start_tasks:
                            break
                        await asyncio.sleep(.005)
                    self.assertTrue(workers.start_tasks)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                    await runtime.prepare_turn(agent, config)
                    state = (await agent.aget_state(config)).values
                    self.assertEqual(len(state['agent_runs']), 1)
                    worker = next(iter(state['agents']))
                    run = next(iter(state['agent_runs']))
                    self.assertEqual(workers.runs[run]['status'], 'running')
                async with OrchestratorRuntime(workspace, settings=settings, data_dir=workspace / 'data') as runtime:
                    agent = runtime.create_agent(ToolCycle())
                    await runtime.prepare_turn(agent, config)
                    state = (await agent.aget_state(config)).values
                    self.assertIn(worker, state['agents'])
                    self.assertIn(run, state['agent_runs'])
                    workers = runtime.workers.runtimes['codex']
                    self.assertEqual(workers.runs[run]['status'], 'interrupted')
                    client = core.get_client(url='http://codex.local')
                    continued = await client.runs.create(worker, 'codex', input={'messages': [{'role': 'user', 'content': 'continue'}]})
                    self.assertEqual(workers.runs[continued['run_id']]['status'], 'success')
                    answer = json.loads(workers.threads[worker]['messages'][-1]['content'])
                    self.assertEqual(answer['previous'], ['slow-start'])
                    self.assertEqual(len(workers.threads), 1)

    async def test_demo_mode_does_not_discard_saved_real_model_context(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            connection = FakeConnection()
            connection.model = 'first'
            with patch('adapters.providers.registry.OpenAIConnection', return_value=connection):
                async with OrchestratorRuntime(workspace, data_dir=workspace / 'data') as runtime:
                    with create_pipe_input() as pipe:
                        ui = ChatUI(runtime.create_agent(DemoCoordinator()), session=runtime.session,
                                    agent_factory=runtime.create_agent, input=pipe, output=DummyOutput())
                        real = ui.config
                        await ui.agent.aupdate_state(real, {'messages': [HumanMessage(content='real context')], 'agents': {}, 'agent_runs': {}}, as_node='tools')
                        connection.model = None
                        ui.activate_model()
                        self.assertNotEqual(ui.config, real)
                        connection.model = 'first'
                        ui.activate_model()
                        self.assertEqual(ui.config, real)
                        self.assertEqual((await ui.agent.aget_state(real)).values['messages'][0].text, 'real context')

    async def test_ui_restores_messages_links_progress_and_stable_thread(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            connection = SimpleNamespace(current_account=lambda: None)
            with patch('adapters.providers.registry.OpenAIConnection', return_value=connection):
                async with OrchestratorRuntime(workspace, data_dir=workspace / 'data') as runtime:
                    with create_pipe_input() as pipe:
                        ui = ChatUI(runtime.create_agent(DemoCoordinator()), session=runtime.session,
                                    prepare_turn=runtime.prepare_turn, input=pipe, output=DummyOutput())
                        ui.add('you', 'remember https://example.com/history')
                        await ui.respond('remember https://example.com/history')
                        thread = ui.config
                        text = ''.join(fragment[1] for fragment in ui.render_transcript())
                async with OrchestratorRuntime(workspace, data_dir=workspace / 'data') as runtime:
                    with create_pipe_input() as pipe:
                        ui = ChatUI(runtime.create_agent(DemoCoordinator()), session=runtime.session,
                                    prepare_turn=runtime.prepare_turn, input=pipe, output=DummyOutput())
                        self.assertEqual(ui.config, thread)
                        self.assertEqual(''.join(fragment[1] for fragment in ui.render_transcript()), text)
                        self.assertTrue(any(fragment[0] == 'class:link' for fragment in ui.fragments))
                        self.assertEqual(ui.progress[-1].stage, 'Responded')
                        await ui.respond('continue')
                        state = (await ui.agent.aget_state(ui.config)).values
                        self.assertEqual(len(state['agents']), 2)
                        export = json.dumps(runtime.session.export())
                        self.assertIn('graph_update', export)
                        self.assertIn('orchestrator_tool', export)


if __name__ == '__main__':
    unittest.main()
