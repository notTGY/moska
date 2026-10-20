"""Assemble model providers, local workers and the unchanged decomposer."""

from contextlib import AsyncExitStack
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from adapters.agents import AgentSettings, LocalWorkers, worker_clients
from decomposer import create_decomposer_agent
from .compaction import ContextCompaction
from .demo import DemoCoordinator, MockWorkers
from .continuity import ToolContinuity
from .storage import SessionStore


class OrchestratorRuntime:
    def __init__(self, workspace=None, *, settings=None, persistent=True, data_dir=None):
        self.workspace = Path(workspace or Path.cwd()).resolve()
        self.settings = settings or AgentSettings()
        self.session = SessionStore(self.workspace, data_dir=data_dir) if persistent else None
        self.workers = None
        self.continuity = None
        self.memory = InMemorySaver()
        self.stack = AsyncExitStack()

    async def __aenter__(self):
        try:
            if self.session:
                self.session.open()
                self.stack.callback(self.session.close)
                self.memory = await self.stack.enter_async_context(
                    AsyncSqliteSaver.from_conn_string(str(self.session.checkpoint_path)))
                await self.memory.setup()
            self.workers = LocalWorkers(self.workspace, self.settings, storage=self.session)
            self.continuity = ToolContinuity(self.session, self.workers)
            mock = MockWorkers(storage=self.session)
            routes = {'http://mock.local': mock.handle, **self.workers.routes}
            await self.stack.enter_async_context(worker_clients(routes))
            return self
        except BaseException:
            await self.stack.aclose()
            raise

    def create_agent(self, model):
        demo = isinstance(model, DemoCoordinator)
        agent_types = ([{'agent_type_id': 'mock', 'description': 'Does nothing and reports "hello".',
                       'assistant_id': 'mock', 'url': 'http://mock.local'}]
                       if demo else self.workers.agent_types)
        return create_decomposer_agent(
            decomposer_model=model, agent_types=agent_types, checkpointer=self.memory,
            agent_run_budget_seconds=None,
            middleware=[self.continuity, *([] if demo else [ContextCompaction(model)])],
        )

    async def prepare_turn(self, agent, config):
        await self.continuity.recover(agent, config)

    async def __aexit__(self, *exc):
        try:
            await self.continuity.settle()
            await self.workers.close()
        finally:
            await self.stack.aclose()


async def chat():
    from tui.app import ChatUI

    async with OrchestratorRuntime() as runtime:
        ui = ChatUI(runtime.create_agent(DemoCoordinator()),
                    agent_factory=runtime.create_agent, worker_runtime=runtime.workers,
                    agent_settings=runtime.settings, session=runtime.session,
                    prepare_turn=runtime.prepare_turn)
        try:
            ui.app.output.write_raw('\x1b[>1u')
            ui.app.output.flush()
            await ui.app.run_async()
        finally:
            ui.app.output.write_raw('\x1b[<u')
            ui.app.output.flush()
