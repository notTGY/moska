import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from collections.abc import Mapping
from typing import NotRequired, TypedDict

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import HumanMessage, SystemMessage

from .prompts import AGENT_RUN_BUDGET_NOTICE, AGENT_GRACEFUL_SHUTDOWN_REQUEST


class RunBudgetContext(TypedDict, total=False):
    agent_run_budget_seconds: float | None
    agent_shutdown_grace_seconds: float


def _run_budget(context):
    if isinstance(context, Mapping):
        return context.get("agent_run_budget_seconds"), context.get("agent_shutdown_grace_seconds", 60.0)
    return getattr(context, "agent_run_budget_seconds", None), getattr(context, "agent_shutdown_grace_seconds", 60.0)


class RunBudgetState(AgentState):
    shutdown_at: NotRequired[float | None]
    timeout_at: NotRequired[float | None]
    shutdown_requested: NotRequired[bool]


def _call_sync_with_timeout(request, handler):
    timeout_at = request.state["timeout_at"]
    if timeout_at is None:
        return handler(request)
    remaining_seconds = timeout_at - time.monotonic()
    if remaining_seconds <= 0:
        raise TimeoutError("Agent run exceeded its work budget and response grace")
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        return executor.submit(copy_context().run, handler, request).result(timeout=remaining_seconds)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


class RunBudgetMiddleware(AgentMiddleware):
    state_schema = RunBudgetState

    def before_agent(self, state, runtime):
        agent_run_budget_seconds, agent_shutdown_grace_seconds = _run_budget(runtime.context)
        shutdown_at = None
        timeout_at = None
        if agent_run_budget_seconds is not None:
            if agent_run_budget_seconds <= 0 or agent_shutdown_grace_seconds <= 0:
                raise ValueError("Run budget and response grace must be positive")
            shutdown_at = time.monotonic() + agent_run_budget_seconds
            timeout_at = shutdown_at + agent_shutdown_grace_seconds
        return {
            "shutdown_at": shutdown_at,
            "timeout_at": timeout_at,
            "shutdown_requested": False,
        }

    async def abefore_agent(self, state, runtime):
        return self.before_agent(state, runtime)

    def before_model(self, state, runtime):
        if state["shutdown_at"] is None:
            return None
        if time.monotonic() >= state["timeout_at"]:
            raise TimeoutError("Agent run exceeded its work budget and response grace")
        if time.monotonic() >= state["shutdown_at"] and not state["shutdown_requested"]:
            return {
                "messages": [HumanMessage(content=AGENT_GRACEFUL_SHUTDOWN_REQUEST)],
                "shutdown_requested": True,
            }
        return None

    def _prepare_model_request(self, request):
        agent_run_budget_seconds, _ = _run_budget(request.runtime.context)
        if agent_run_budget_seconds is not None:
            notice = AGENT_RUN_BUDGET_NOTICE.format(
                agent_run_budget_seconds=agent_run_budget_seconds,
            )
            system_message = request.system_message
            if system_message is None:
                system_message = SystemMessage(content=notice)
            else:
                content = system_message.content
                content = content + "\n\n" + notice if isinstance(content, str) else [
                    *content, {"type": "text", "text": notice},
                ]
                system_message = system_message.model_copy(update={"content": content})
            request = request.override(system_message=system_message)
        if request.state["shutdown_requested"]:
            request = request.override(tools=[], tool_choice=None, response_format=None)
        return request

    async def awrap_model_call(self, request, handler):
        request = self._prepare_model_request(request)
        timeout_at = request.state["timeout_at"]
        remaining_seconds = None if timeout_at is None else max(0, timeout_at - time.monotonic())
        async with asyncio.timeout(remaining_seconds):
            return await handler(request)

    async def awrap_tool_call(self, request, handler):
        timeout_at = request.state["timeout_at"]
        remaining_seconds = None if timeout_at is None else max(0, timeout_at - time.monotonic())
        async with asyncio.timeout(remaining_seconds):
            return await handler(request)

    def wrap_model_call(self, request, handler):
        request = self._prepare_model_request(request)
        return _call_sync_with_timeout(request, handler)

    def wrap_tool_call(self, request, handler):
        return _call_sync_with_timeout(request, handler)

    def after_model(self, state, runtime):
        if state["timeout_at"] is not None and time.monotonic() >= state["timeout_at"]:
            raise TimeoutError("Agent run exceeded its work budget and response grace")
        if state["shutdown_requested"] and state["messages"][-1].tool_calls:
            raise ValueError("Agent requested tools instead of returning its report")
        return None
