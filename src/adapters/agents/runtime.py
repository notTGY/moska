"""Public local-worker runtime API, independent of orchestrator model providers."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import patch

import httpx
from langgraph_sdk.client import LangGraphClient

from .registry import AGENT_SPECS


class LocalWorkers:
    def __init__(self, workspace, settings, *, specs=AGENT_SPECS):
        self.on_event = lambda agent, item: None
        self.runtimes = {
            spec.agent_id: spec.runtime_factory(
                workspace, settings, on_event=lambda agent, item: self.on_event(agent, item),
            )
            for spec in specs if spec.runtime_factory is not None
        }

    @property
    def agent_types(self):
        return [runtime.agent_type for runtime in self.runtimes.values()]

    @property
    def routes(self):
        return {runtime.agent_type['url']: runtime.handle for runtime in self.runtimes.values()}

    async def interrupt_all(self):
        await asyncio.gather(*(runtime.interrupt_all() for runtime in self.runtimes.values()))

    async def close(self):
        await asyncio.gather(*(runtime.close() for runtime in self.runtimes.values()))


@asynccontextmanager
async def worker_clients(routes):
    """Compatibility seam for the external decomposer's HTTP client factory.

    Requests are dispatched entirely in process. Only registered local URLs are
    accepted; no auth headers or API keys are loaded by the LangGraph factory.
    """
    from contextlib import AsyncExitStack

    async with AsyncExitStack() as stack:
        clients = {}
        for url, handler in routes.items():
            http = await stack.enter_async_context(httpx.AsyncClient(
                transport=httpx.MockTransport(handler), base_url=url,
            ))
            clients[url] = LangGraphClient(http)

        def local_client(*, url=None, **kwargs):
            if url not in clients:
                raise ValueError('Only registered in-process workers are supported.')
            return clients[url]

        # The external decomposer has no public client-injection parameter.
        # Scope the factory override to the owning application runtime.
        with patch('decomposer.core.get_client', side_effect=local_client), patch(
            'decomposer.core.get_sync_client',
            side_effect=RuntimeError('Local workers require async invocation.'),
        ):
            yield
