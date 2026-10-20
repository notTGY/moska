import asyncio
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from adapters.agents.bridge import CodexWorkers
from adapters.agents.codex import CodexError, CodexServer
from adapters.agents.registry import AGENT_SPECS, AgentSettings, DetectedAgent, scan_agents
from orchestrator.runtime import OrchestratorRuntime
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from tui.app import ChatUI

FAKE_CODEX = str(Path(__file__).with_name('fake_codex.py').resolve())


class ToolCycle(BaseChatModel):
    calls: list = Field(default_factory=list)
    worker_prompt: str = 'inspect workspace'

    @property
    def _llm_type(self):
        return 'codex-bridge-test'

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        last = messages[-1]
        self.calls.append(last)
        if isinstance(last, HumanMessage):
            if any(isinstance(message, ToolMessage) and message.name == 'run' for message in messages):
                name, args = 'wait', {}
            else:
                name, args = 'new', {'agent_type_id': 'codex'}
        else:
            output = json.loads(last.content)
            if last.name == 'new':
                name, args = 'run', {'agent_id': output['agent_id'], 'prompt': self.worker_prompt}
            elif last.name == 'run':
                name, args = 'wait', {}
            else:
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=output[0]['response']))])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content='', tool_calls=[{'id': str(uuid4()), 'name': name, 'args': args}]))])


class CodexVersionTests(unittest.TestCase):
    def test_rejects_old_unknown_and_prerelease_baseline(self):
        for version in ('0.162.0', '0.163.0-alpha.1', None, 'invalid'):
            with self.subTest(version=version):
                entry = DetectedAgent(AGENT_SPECS[0], '/bin/codex', version)
                self.assertFalse(entry.available)
                self.assertIn('0.163.0', entry.hint)
        for version in ('0.163.0', '0.164.0', '1.0.0', '0.163.0+build'):
            self.assertTrue(DetectedAgent(AGENT_SPECS[0], '/bin/codex', version).available)

    def test_version_probe_failure_disables_entry(self):
        for error in (subprocess.TimeoutExpired('codex', 3), OSError('missing')):
            with patch('adapters.agents.registry.shutil.which', return_value='/bin/codex'), patch('adapters.agents.registry.subprocess.run', side_effect=error):
                entry = scan_agents()[0]
                self.assertFalse(entry.available)
                self.assertIn('version unknown', entry.hint)


class CodexServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_handshake_denies_approvals_and_unknown_host_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            events = []
            server = CodexServer(FAKE_CODEX, Path(directory), on_event=lambda method, params: events.append(method))
            try:
                await server.start()
                await server.request('test/approval', {})
                for _ in range(100):
                    if 'test/unknownDenied' in events:
                        break
                    await asyncio.sleep(.01)
                self.assertIn('test/denied', events)
                self.assertIn('test/unknownDenied', events)
                self.assertTrue(server.ready)
            finally:
                await server.close()
            self.assertIsNotNone(server.process.returncode)
            self.assertFalse(server.pending)

    async def test_timeout_stops_server_instead_of_leaving_unknown_work(self):
        with tempfile.TemporaryDirectory() as directory:
            server = CodexServer(FAKE_CODEX, Path(directory))
            try:
                await server.start()
                with self.assertRaises(CodexError):
                    await server.request('test/no-response', {}, timeout=.05)
                self.assertIsNotNone(server.process.returncode)
                self.assertFalse(server.ready)
            finally:
                await server.close()


class CodexBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        self.settings = AgentSettings(self.workspace / 'settings.json')
        self.settings.save({'codex'})
        self.version = patch('adapters.agents.bridge.require_codex', return_value=FAKE_CODEX)
        self.version.start()
        self.runtime = await OrchestratorRuntime(self.workspace, settings=self.settings, persistent=False).__aenter__()
        self.workers = self.runtime.workers.runtimes['codex']
        self.events = []
        self.runtime.workers.on_event = lambda agent, item: self.events.append((agent, item))
        import httpx
        from langgraph_sdk.client import LangGraphClient
        self.http = httpx.AsyncClient(transport=httpx.MockTransport(self.workers.handle), base_url='http://codex.local')
        self.client = LangGraphClient(self.http)

    async def asyncTearDown(self):
        await self.http.aclose()
        await self.runtime.__aexit__(None, None, None)
        self.version.stop()
        self.temporary.cleanup()

    async def new(self):
        return (await self.client.threads.create())['thread_id']

    async def start_run(self, thread, prompt):
        return await self.client.runs.create(thread_id=thread, assistant_id='codex', input={'messages': [{'role': 'user', 'content': prompt}]})

    async def test_native_fork_continuation_and_workspace(self):
        thread = await self.new()
        first = await self.start_run(thread, 'first')
        self.assertEqual(first['status'], 'pending')
        self.assertEqual((await self.client.runs.get(thread, first['run_id']))['status'], 'success')
        fork = (await self.client.threads.copy(thread))['thread_id']
        self.assertNotEqual(thread, fork)
        await self.start_run(fork, 'branch')
        await self.start_run(thread, 'continue')
        self.assertEqual(json.loads(self.workers.threads[fork]['messages'][-1]['content'])['previous'], ['first'])
        self.assertEqual(json.loads(self.workers.threads[thread]['messages'][-1]['content'])['previous'], ['first'])
        history = await self.client.threads.get_history(thread, metadata={'run_id': first['run_id']})
        final = json.loads(history[0]['values']['messages'][-1]['content'])
        self.assertEqual(final['cwd'], str(self.workspace.resolve()))
        self.assertEqual(history[-1]['metadata']['source'], 'input')
        self.assertEqual(history[0]['values']['messages'][-3]['tool_calls'][0]['name'], 'commandExecution')
        self.assertTrue(any(item['type'] == 'commandExecution' for _, item in self.events))

    async def test_parallel_agents_interrupt_and_reuse(self):
        first, second = await asyncio.gather(self.new(), self.new())
        runs = await asyncio.gather(self.start_run(first, 'hold'), self.start_run(second, 'hold'))
        self.assertEqual(len(self.workers.active), 2)
        self.assertTrue(self.workers.server.ready)
        with self.assertRaises(Exception):
            await self.start_run(first, 'duplicate')
        await self.workers.interrupt_all()
        self.assertFalse(self.workers.active)
        self.assertEqual([self.workers.runs[run['run_id']]['status'] for run in runs], ['interrupted'] * 2)
        run = await self.start_run(first, 'after interruption')
        self.assertEqual(self.workers.runs[run['run_id']]['status'], 'success')

    async def test_interruption_during_start_does_not_orphan_worker(self):
        thread = await self.new()
        task = asyncio.create_task(self.start_run(thread, 'slow-start'))
        for _ in range(100):
            if self.workers.start_tasks:
                break
            await asyncio.sleep(.005)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await self.workers.interrupt_all()
        self.assertFalse(self.workers.active)
        self.assertEqual(next(iter(self.workers.runs.values()))['status'], 'interrupted')

    async def test_interrupt_retries_when_codex_acknowledges_before_turn_is_active(self):
        thread = await self.new()
        run = await self.start_run(thread, 'queued')
        await self.workers.interrupt_all()
        self.assertEqual(self.workers.runs[run['run_id']]['status'], 'interrupted')
        self.assertTrue(self.workers.server.ready)
        self.assertFalse(self.workers.active)

    async def test_errors_and_disconnect_are_terminal(self):
        thread = await self.new()
        failed = await self.start_run(thread, 'fail')
        self.assertEqual(self.workers.runs[failed['run_id']]['status'], 'error')
        self.assertEqual((await self.client.threads.get(thread))['error']['message'], 'worker failure')
        crashed = await self.start_run(thread, 'crash')
        self.assertEqual(self.workers.runs[crashed['run_id']]['status'], 'error')
        self.assertFalse(self.workers.active)

    async def test_disabled_runtime_never_launches_and_version_checked_again(self):
        self.settings.save(set())
        with self.assertRaises(Exception):
            await self.new()
        self.assertIsNone(self.workers.server)
        self.settings.save({'codex'})
        with patch('adapters.agents.bridge.require_codex', side_effect=CodexError('requires 0.163.0')):
            with self.assertRaisesRegex(Exception, '0.163.0'):
                await self.new()
        self.assertIsNone(self.workers.server)

    async def test_original_decomposer_collects_real_worker_messages_and_tools(self):
        model = ToolCycle()
        agent = self.runtime.create_agent(model)
        state = await agent.ainvoke({'messages': [HumanMessage(content='inspect')], 'agents': {}, 'agent_runs': {}}, {'configurable': {'thread_id': 'test'}})
        final = json.loads(state['messages'][-1].content)
        self.assertEqual(final['cwd'], str(self.workspace.resolve()))
        report = next(iter(state['agent_runs'].values()))
        self.assertEqual(report['status'], 'responded')
        self.assertEqual(len(report['tool_calls']), 1)
        self.assertEqual(report['tool_calls'][0]['name'], 'commandExecution')
        self.assertEqual([message.name for message in model.calls if isinstance(message, ToolMessage)], ['new', 'run', 'wait'])

    async def test_followups_steer_the_orchestrator_without_restarting_workers(self):
        model = ToolCycle(worker_prompt='hold')
        agent = self.runtime.create_agent(model)
        disconnected = SimpleNamespace(current_account=lambda: None)
        with patch('adapters.providers.registry.OpenAIConnection', return_value=disconnected), create_pipe_input() as pipe:
            ui = ChatUI(agent, worker_runtime=self.runtime.workers, prepare_turn=self.runtime.prepare_turn,
                        input=pipe, output=DummyOutput())
            task = asyncio.create_task(ui.app.run_async())
            try:
                await asyncio.sleep(.05)
                pipe.send_text('inspect\r')
                for _ in range(200):
                    if model.calls and isinstance(model.calls[-1], ToolMessage) and model.calls[-1].name == 'run':
                        break
                    await asyncio.sleep(.01)
                worker = next(iter(self.workers.active))
                config = ui.config
                pipe.send_text('focus on tests\r')
                for _ in range(200):
                    if len(ui.progress) > 1 and any(isinstance(message, HumanMessage) and message.content == 'focus on tests' for message in model.calls):
                        break
                    await asyncio.sleep(.01)
                self.assertEqual(ui.progress[0].stage, 'Steered')
                self.assertIn(worker, self.workers.active)
                self.assertEqual(ui.config, config)
                self.assertEqual(len(self.workers.threads), 1)
                self.assertIn('1 agent running', ui.worker_counter)
                pipe.send_text('/ps\r')
                await asyncio.sleep(.05)
                view = ''.join(fragment[1] for fragment in ui.render_transcript())
                self.assertIn('Workspace: ' + str(self.workspace.resolve()), view)
                self.assertIn('Prompt: hold', view)
                self.assertIn(worker, view)
                await self.workers.server.request('test/complete', {'threadId': worker})
                for _ in range(200):
                    if ui.progress[-1].finished:
                        break
                    await asyncio.sleep(.01)
                self.assertEqual(ui.progress[-1].stage, 'Responded')
                self.assertIn('0 agents running', ui.worker_counter)
                pipe.send_text('/exit\r')
                await asyncio.wait_for(task, 2)
            finally:
                if not task.done():
                    ui.app.exit()
                    await task

    async def test_stop_command_stops_all_agents_while_orchestrator_is_running(self):
        agent = self.runtime.create_agent(ToolCycle(worker_prompt='hold'))
        disconnected = SimpleNamespace(current_account=lambda: None)
        with patch('adapters.providers.registry.OpenAIConnection', return_value=disconnected), create_pipe_input() as pipe:
            ui = ChatUI(agent, worker_runtime=self.runtime.workers, prepare_turn=self.runtime.prepare_turn,
                        input=pipe, output=DummyOutput())
            task = asyncio.create_task(ui.app.run_async())
            try:
                await asyncio.sleep(.05)
                pipe.send_text('inspect\r')
                for _ in range(200):
                    if self.workers.active:
                        break
                    await asyncio.sleep(.01)
                self.assertEqual(self.runtime.workers.running_count, 1)
                pipe.send_text('/stop\r')
                for _ in range(200):
                    if ui.stop_task and ui.stop_task.done():
                        break
                    await asyncio.sleep(.01)
                self.assertFalse(self.workers.active)
                self.assertTrue(ui.task.cancelled())
                self.assertIn('Stopped 1 background agent.', ''.join(fragment[1] for fragment in ui.render_transcript()))
                self.assertFalse(self.workers.stopping)
                self.assertTrue(self.workers.server.ready)
                pipe.send_text('/exit\r')
                await asyncio.wait_for(task, 2)
            finally:
                if not task.done():
                    ui.app.exit()
                    await task

    async def test_ui_escape_preserves_workers_context_and_later_collects_results(self):
        agent = self.runtime.create_agent(ToolCycle(worker_prompt='hold'))
        disconnected = SimpleNamespace(current_account=lambda: None)
        with patch('adapters.providers.registry.OpenAIConnection', return_value=disconnected), create_pipe_input() as pipe:
            ui = ChatUI(agent, worker_runtime=self.runtime.workers, prepare_turn=self.runtime.prepare_turn,
                        input=pipe, output=DummyOutput())
            config = ui.config
            task = asyncio.create_task(ui.app.run_async())
            try:
                await asyncio.sleep(.05)
                pipe.send_text('inspect\r')
                for _ in range(200):
                    if self.workers.active and self.events:
                        break
                    await asyncio.sleep(.01)
                self.assertTrue(self.workers.active)
                expanded = ''.join(fragment[1] for fragment in ui.progress[-1].render(True))
                self.assertIn('run', expanded)
                self.assertNotIn('command', expanded)
                self.assertNotIn('pwd', expanded)
                pipe.send_bytes(b'\x1b')
                for _ in range(200):
                    if ui.task.done():
                        break
                    await asyncio.sleep(.01)
                self.assertTrue(ui.task.cancelled())
                self.assertTrue(self.workers.active)
                self.assertEqual(ui.config, config)
                self.assertEqual(ui.progress[-1].stage, 'Interrupted')
                self.assertFalse(task.done())
                worker = next(iter(self.workers.active))
                await self.workers.server.request('test/complete', {'threadId': worker})
                pipe.send_text('collect the result\r')
                for _ in range(200):
                    if len(ui.progress) == 2 and ui.progress[-1].finished:
                        break
                    await asyncio.sleep(.01)
                self.assertEqual(ui.progress[-1].stage, 'Responded')
                self.assertIn('background result', ''.join(fragment[1] for fragment in ui.render_transcript()))
                self.assertEqual(len(self.workers.threads), 1)
                pipe.send_text('/exit\r')
                await asyncio.wait_for(task, 2)
            finally:
                if not task.done():
                    ui.app.exit()
                    await task
