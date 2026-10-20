import json
import time
import asyncio
import logging
from collections import Counter
from dataclasses import asdict, is_dataclass
from typing import Annotated, Any, NotRequired, Sequence
from uuid import uuid4

from httpx import ReadError, RemoteProtocolError
from langchain.agents import create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ResponseT,
)
from langchain.tools import ToolRuntime
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import StructuredTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Checkpointer, Command
from langgraph_sdk import get_client, get_sync_client
from langgraph_sdk.client import LangGraphClient, SyncLangGraphClient
from langsmith import trace
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from .prompts import (
    NEW_TOOL_DESCRIPTION,
    FORK_TOOL_DESCRIPTION,
    PARALLEL_FORK_RUN_CALL_ERROR,
    DECOMPOSER_SYSTEM_PROMPT,
    RUN_BUDGET_CONVENTION,
    PARALLEL_RUN_CALL_ERROR,
    EARLY_RESPONSE_ERROR,
    EMPTY_PROMPT_ERROR,
    DUPLICATE_PROMPT_ERROR,
    DECOMPOSER_GRACEFUL_SHUTDOWN_REQUEST,
    EMPTY_AGENT_RESPONSE_ERROR,
    EMPTY_RESPONSE_ERROR,
    FAILED_RUN_ERROR,
    WAIT_TIMEOUT_ERROR,
    NO_ACTIVE_RUNS_ERROR,
    ACTIVE_RUN_ERROR,
    PARALLEL_WAIT_CALL_ERROR,
    PROMPT_PARAMETER_DESCRIPTION,
    RUN_TOOL_DESCRIPTION,
    AGENT_ID_PARAMETER_DESCRIPTION,
    AGENT_TYPE_ID_PARAMETER_DESCRIPTION,
    UNKNOWN_AGENT_ERROR,
    UNKNOWN_AGENT_TYPE_ERROR,
    WAIT_TOOL_DESCRIPTION,
)

logger = logging.getLogger(__name__)


# AGENT_PROMPT_MAX_TOKENS = 1024
# AGENT_RESPONSE_MAX_TOKENS = 1024
WAIT_TIMEOUT_SECONDS = 60.0
WAIT_POLL_SECONDS = 1.0
TERMINAL_STATUSES = frozenset({"responded", "error", "timeout", "interrupted"})
HISTORY_LIMIT = 1000


class AgentType(TypedDict):
    agent_type_id: str
    description: str
    assistant_id: str
    url: NotRequired[str]
    headers: NotRequired[dict[str, str]]


class AgentToolCall(TypedDict):
    id: str
    name: str
    args: dict[str, Any]


class Agent(TypedDict):
    agent_id: str
    agent_type_id: str
    assistant_id: str
    thread_id: str
    created_at: float  # Unix timestamp in seconds.
    forked_from: NotRequired[str]


class AgentRun(TypedDict):
    agent_run_id: str
    agent_id: str
    run_id: str
    status: str
    prompt: str
    started_at: float  # Unix timestamp when starting the run.
    # Unix timestamp when returning the result: from wait() or the invocation.
    collected_at: NotRequired[float]
    messages: NotRequired[list[dict[str, Any]]]
    tool_calls: NotRequired[list[AgentToolCall]]
    response: NotRequired[str | None]
    # Zero-based order in which wait() returned this run's response.
    response_sequence_number: NotRequired[int]
    error: NotRequired[str | None]


def _agents_reducer(
    existing: dict[str, Agent] | None,
    update: dict[str, Agent],
) -> dict[str, Agent]:
    merged = dict(existing or {})
    merged.update(update)
    return merged


def _agent_runs_reducer(
    existing: dict[str, AgentRun] | None,
    update: dict[str, AgentRun],
) -> dict[str, AgentRun]:
    merged = dict(existing or {})
    merged.update(update)
    return merged


def _normalize_run_status(status: str) -> str:
    return "responded" if status == "success" else status


def _get_current_agent_runs(
    agent_runs: dict[str, AgentRun],
) -> dict[str, AgentRun]:
    terminal_runs_without_responses = {
        agent_run_id: agent_run
        for agent_run_id, agent_run in agent_runs.items()
        if agent_run["status"] in TERMINAL_STATUSES
        and "response_sequence_number" not in agent_run
    }
    if terminal_runs_without_responses:
        details = ", ".join(
            f"`{agent_run_id}` ({agent_run['status']})"
            for agent_run_id, agent_run in terminal_runs_without_responses.items()
        )
        raise RuntimeError(
            "Invalid Decomposer state: terminal agent runs have no collected "
            f"response: {details}."
        )

    return {
        agent_run_id: agent_run
        for agent_run_id, agent_run in agent_runs.items()
        if agent_run["status"] not in TERMINAL_STATUSES
    }


def _build_new_schema(
    agent_types: dict[str, AgentType],
) -> type[BaseModel]:
    available_agent_types = "\n".join(
        f"| {agent_type_id} | {agent_type['description']} |"
        for agent_type_id, agent_type in agent_types.items()
    )
    description = AGENT_TYPE_ID_PARAMETER_DESCRIPTION.format(
        available_agent_types=available_agent_types
    )

    class NewSchema(BaseModel):
        agent_type_id: str = Field(description=description)

    return NewSchema


class ForkSchema(BaseModel):
    agent_id: str = Field(description=AGENT_ID_PARAMETER_DESCRIPTION)


class RunSchema(BaseModel):
    agent_id: str = Field(description=AGENT_ID_PARAMETER_DESCRIPTION)
    prompt: str = Field(description=PROMPT_PARAMETER_DESCRIPTION)


class DecomposerAgentState(AgentState[ResponseT]):
    run_prompt_counts: NotRequired[dict[str, int]]
    shutdown_requested: NotRequired[bool]
    decomposer_agent_runs: NotRequired[list[AgentRun]]
    agents: Annotated[
        NotRequired[dict[str, Agent]], _agents_reducer
    ]
    agent_runs: Annotated[
        NotRequired[dict[str, AgentRun]], _agent_runs_reducer
    ]


def _count_text_tokens(text: str) -> int:
    return count_tokens_approximately([{"role": "user", "content": text}])


def _truncate_text(text: str | None, max_tokens: int) -> tuple[str | None, bool]:
    if text is None or _count_text_tokens(text) <= max_tokens:
        return text, False

    suffix = f"\n\n[truncated to approximately {max_tokens} tokens]"
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = text[:mid].rstrip() + suffix
        if _count_text_tokens(candidate) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + suffix, True


def _extract_agent_tool_calls(messages: list[dict[str, Any]]) -> list[AgentToolCall]:
    tool_calls: list[AgentToolCall] = []
    for message in messages:
        if message["type"] != "ai":
            continue
        for tool_call in message["tool_calls"]:
            tool_call_id = tool_call["id"]
            name = tool_call["name"]
            args = tool_call["args"]
            if not isinstance(tool_call_id, str) or not isinstance(name, str) or not isinstance(args, dict):
                raise ValueError(f"Invalid agent tool call in run history: {tool_call!r}")
            tool_calls.append({"id": tool_call_id, "name": name, "args": args})
    return tool_calls


def _resolve_headers(agent_type: AgentType) -> dict[str, str]:
    headers: dict[str, str] = dict(agent_type.get("headers") or {})
    if "x-auth-scheme" not in headers:
        headers["x-auth-scheme"] = "langsmith"
    return headers


class _ClientCache:
    """Adapted from deepagents.middleware.async_subagents.ClientCache."""

    def __init__(self, agent_types: dict[str, AgentType]) -> None:
        self._agent_types = agent_types
        self._sync: dict[
            tuple[str | None, frozenset[tuple[str, str]]], SyncLangGraphClient
        ] = {}
        self._async: dict[
            tuple[str | None, frozenset[tuple[str, str]]], LangGraphClient
        ] = {}

    def _cache_key(
        self, agent_type: AgentType
    ) -> tuple[str | None, frozenset[tuple[str, str]]]:
        return (
            agent_type.get("url"),
            frozenset(_resolve_headers(agent_type).items()),
        )

    def get_sync(self, agent_type_id: str) -> SyncLangGraphClient:
        agent_type = self._agent_types[agent_type_id]
        if agent_type.get("url") is None:
            msg = f"Agent type '{agent_type_id}' has no url configured. ASGI transport (url=None) requires async invocation."
            raise ValueError(msg)
        key = self._cache_key(agent_type)
        if key not in self._sync:
            self._sync[key] = get_sync_client(
                url=agent_type.get("url"),
                headers=_resolve_headers(agent_type),
            )
        return self._sync[key]

    def get_async(self, agent_type_id: str) -> LangGraphClient:
        agent_type = self._agent_types[agent_type_id]
        key = self._cache_key(agent_type)
        if key not in self._async:
            self._async[key] = get_client(
                url=agent_type.get("url"),
                headers=_resolve_headers(agent_type),
            )
        return self._async[key]


def _build_new_tool(
    agent_types: dict[str, AgentType],
    clients: _ClientCache,
) -> StructuredTool:

    def new(
        agent_type_id: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        if agent_type_id not in agent_types:
            allowed = ", ".join(f"`{k}`" for k in agent_types)
            return UNKNOWN_AGENT_TYPE_ERROR.format(
                agent_type_id=agent_type_id, allowed=allowed
            )

        agent_type = agent_types[agent_type_id]
        client = clients.get_sync(agent_type_id)
        with trace("threads.create", metadata={"agent_type_id": agent_type_id}):
            thread = client.threads.create()
        agent_id = thread["thread_id"]
        logger.debug(
            "Created agent: tool_call_id=%s agent_id=%s type=%s",
            runtime.tool_call_id, agent_id, agent_type_id,
        )
        agent: Agent = {
            "agent_id": agent_id,
            "agent_type_id": agent_type_id,
            "assistant_id": agent_type["assistant_id"],
            "thread_id": thread["thread_id"],
            "created_at": time.time(),
        }
        tool_output: dict[str, Any] = {
            "agent_id": agent_id,
        }
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        json.dumps(tool_output, ensure_ascii=False),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
                "agents": {agent_id: agent},
            }
        )

    async def anew(
        agent_type_id: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        if agent_type_id not in agent_types:
            allowed = ", ".join(f"`{k}`" for k in agent_types)
            return UNKNOWN_AGENT_TYPE_ERROR.format(
                agent_type_id=agent_type_id, allowed=allowed
            )

        agent_type = agent_types[agent_type_id]
        client = clients.get_async(agent_type_id)
        with trace("threads.create", metadata={"agent_type_id": agent_type_id}):
            thread = await client.threads.create()
        agent_id = thread["thread_id"]
        logger.debug(
            "Created agent: tool_call_id=%s agent_id=%s type=%s",
            runtime.tool_call_id, agent_id, agent_type_id,
        )
        agent: Agent = {
            "agent_id": agent_id,
            "agent_type_id": agent_type_id,
            "assistant_id": agent_type["assistant_id"],
            "thread_id": thread["thread_id"],
            "created_at": time.time(),
        }
        tool_output: dict[str, Any] = {
            "agent_id": agent_id,
        }
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        json.dumps(tool_output, ensure_ascii=False),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
                "agents": {agent_id: agent},
            }
        )

    return StructuredTool.from_function(
        func=new,
        coroutine=anew,
        name="new",
        description=NEW_TOOL_DESCRIPTION,
        infer_schema=False,
        args_schema=_build_new_schema(agent_types),
    )


def _get_agent_error(
    agent_id: str,
    state: DecomposerAgentState,
) -> str | None:
    if agent_id not in state["agents"]:
        return UNKNOWN_AGENT_ERROR.format(agent_id=agent_id)

    for agent_run_id, agent_run in state["agent_runs"].items():
        if agent_run["agent_id"] != agent_id:
            continue
        if "response_sequence_number" not in agent_run:
            return ACTIVE_RUN_ERROR.format(
                agent_id=agent_id, agent_run_id=agent_run_id
            )
        if agent_run["status"] != "responded":
            return FAILED_RUN_ERROR.format(
                agent_id=agent_id,
                agent_run_id=agent_run_id,
                status=agent_run["status"],
            )
    return None


def _build_fork_tool(clients: _ClientCache) -> StructuredTool:

    def fork(agent_id: str, runtime: ToolRuntime) -> str | Command:
        error = _get_agent_error(agent_id, runtime.state)
        if error is not None:
            return error

        source = runtime.state["agents"][agent_id]
        client = clients.get_sync(source["agent_type_id"])
        with trace("threads.get_state", metadata={"thread_id": source["thread_id"]}):
            state = client.threads.get_state(source["thread_id"])
        # Copying a thread without checkpoints breaks the in-memory runtime.
        if state["checkpoint"] is None:
            with trace("threads.create"):
                thread = client.threads.create()
        else:
            with trace("threads.copy", metadata={"thread_id": source["thread_id"]}):
                thread = client.threads.copy(source["thread_id"])
        agent_id = thread["thread_id"]
        logger.debug(
            "Forked agent: tool_call_id=%s source=%s agent_id=%s",
            runtime.tool_call_id, source["agent_id"], agent_id,
        )
        agent: Agent = {
            **source,
            "agent_id": agent_id,
            "thread_id": thread["thread_id"],
            "created_at": time.time(),
            "forked_from": source["agent_id"],
        }
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        json.dumps({"agent_id": agent_id}, ensure_ascii=False),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
                "agents": {agent_id: agent},
            }
        )

    async def afork(agent_id: str, runtime: ToolRuntime) -> str | Command:
        error = _get_agent_error(agent_id, runtime.state)
        if error is not None:
            return error

        source = runtime.state["agents"][agent_id]
        client = clients.get_async(source["agent_type_id"])
        with trace("threads.get_state", metadata={"thread_id": source["thread_id"]}):
            state = await client.threads.get_state(source["thread_id"])
        # Copying a thread without checkpoints breaks the in-memory runtime.
        if state["checkpoint"] is None:
            with trace("threads.create"):
                thread = await client.threads.create()
        else:
            with trace("threads.copy", metadata={"thread_id": source["thread_id"]}):
                thread = await client.threads.copy(source["thread_id"])
        agent_id = thread["thread_id"]
        logger.debug(
            "Forked agent: tool_call_id=%s source=%s agent_id=%s",
            runtime.tool_call_id, source["agent_id"], agent_id,
        )
        agent: Agent = {
            **source,
            "agent_id": agent_id,
            "thread_id": thread["thread_id"],
            "created_at": time.time(),
            "forked_from": source["agent_id"],
        }
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        json.dumps({"agent_id": agent_id}, ensure_ascii=False),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
                "agents": {agent_id: agent},
            }
        )

    return StructuredTool.from_function(
        func=fork,
        coroutine=afork,
        name="fork",
        description=FORK_TOOL_DESCRIPTION,
        infer_schema=False,
        args_schema=ForkSchema,
    )


def _get_run_prompt_error(prompt: str, state: DecomposerAgentState) -> str | None:
    if not prompt.strip():
        return EMPTY_PROMPT_ERROR
    if any(
        run["prompt"] == prompt
        and run["started_at"] >= state["decomposer_agent_runs"][-1]["started_at"]
        for run in state["agent_runs"].values()
    ):
        return DUPLICATE_PROMPT_ERROR
    return None


def _build_run_tool(
    clients: _ClientCache,
    recursion_limit: int | None,
    agent_run_budget_seconds: float | None = None,
    agent_shutdown_grace_seconds: float = 60.0,
) -> StructuredTool:
    run_config = {}
    if recursion_limit is not None:
        run_config["recursion_limit"] = recursion_limit

    def run_context(context):
        if agent_run_budget_seconds is None:
            return context
        if is_dataclass(context):
            context = asdict(context)
        elif isinstance(context, BaseModel):
            context = context.model_dump()
        return {
            **(context or {}),
            "agent_run_budget_seconds": agent_run_budget_seconds,
            "agent_shutdown_grace_seconds": agent_shutdown_grace_seconds,
        }

    def run(
        agent_id: str,
        prompt: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        error = _get_run_prompt_error(prompt, runtime.state)
        if error is not None:
            return error

        error = _get_agent_error(agent_id, runtime.state)
        if error is not None:
            return error

        # prompt_token_count = _count_text_tokens(prompt)
        # if prompt_token_count > AGENT_PROMPT_MAX_TOKENS:
        #     return f"The prompt is too long (about {prompt_token_count} tokens) while the limit is {AGENT_PROMPT_MAX_TOKENS} tokens."

        agent = runtime.state["agents"][agent_id]
        client = clients.get_sync(agent["agent_type_id"])
        started_at = time.time()
        with trace("runs.create", metadata={"agent_id": agent_id, "thread_id": agent["thread_id"]}):
            run = client.runs.create(
                thread_id=agent["thread_id"],
                assistant_id=agent["assistant_id"],
                input={"messages": [{"role": "user", "content": prompt}]},
                config=run_config or None,
                context=run_context(runtime.context),
                multitask_strategy="reject",
            )

        agent_run_id = run["run_id"]
        logger.info(
            "Started run: tool_call_id=%s agent_id=%s run_id=%s status=%s",
            runtime.tool_call_id, agent_id, agent_run_id, run["status"],
        )
        status = _normalize_run_status(run["status"])
        if status in TERMINAL_STATUSES:
            raise ValueError(
                f"`client.runs.create` returned a run `{agent_run_id}` with terminal status `{status}`."
            )
        agent_run: AgentRun = {
            "agent_run_id": agent_run_id,
            "agent_id": agent_id,
            "run_id": run["run_id"],
            "status": status,
            "prompt": prompt,
            "started_at": started_at,
        }
        tool_output: dict[str, Any] = {
            "agent_run_id": agent_run_id,
        }
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        json.dumps(tool_output, ensure_ascii=False),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
                "agent_runs": {agent_run_id: agent_run},
            }
        )

    async def arun(
        agent_id: str,
        prompt: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        error = _get_run_prompt_error(prompt, runtime.state)
        if error is not None:
            return error

        error = _get_agent_error(agent_id, runtime.state)
        if error is not None:
            return error

        # prompt_token_count = _count_text_tokens(prompt)
        # if prompt_token_count > AGENT_PROMPT_MAX_TOKENS:
        #     return f"The prompt is too long (about {prompt_token_count} tokens) while the limit is {AGENT_PROMPT_MAX_TOKENS} tokens."

        agent = runtime.state["agents"][agent_id]
        client = clients.get_async(agent["agent_type_id"])
        started_at = time.time()
        with trace("runs.create", metadata={"agent_id": agent_id, "thread_id": agent["thread_id"]}):
            run = await client.runs.create(
                thread_id=agent["thread_id"],
                assistant_id=agent["assistant_id"],
                input={"messages": [{"role": "user", "content": prompt}]},
                config=run_config or None,
                context=run_context(runtime.context),
                multitask_strategy="reject",
            )

        agent_run_id = run["run_id"]
        logger.info(
            "Started run: tool_call_id=%s agent_id=%s run_id=%s status=%s",
            runtime.tool_call_id, agent_id, agent_run_id, run["status"],
        )
        status = _normalize_run_status(run["status"])
        if status in TERMINAL_STATUSES:
            raise ValueError(
                f"`client.runs.create` returned a run `{agent_run_id}` with terminal status `{status}`."
            )
        agent_run: AgentRun = {
            "agent_run_id": agent_run_id,
            "agent_id": agent_id,
            "run_id": run["run_id"],
            "status": status,
            "prompt": prompt,
            "started_at": started_at,
        }
        tool_output: dict[str, Any] = {
            "agent_run_id": agent_run_id,
        }
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        json.dumps(tool_output, ensure_ascii=False),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
                "agent_runs": {agent_run_id: agent_run},
            }
        )

    return StructuredTool.from_function(
        func=run,
        coroutine=arun,
        name="run",
        description=RUN_TOOL_DESCRIPTION,
        infer_schema=False,
        args_schema=RunSchema,
    )


def _build_wait_tool(
    clients: _ClientCache,
) -> StructuredTool:

    def wait(runtime: ToolRuntime) -> str | Command:
        agents: dict[str, Agent] = runtime.state["agents"]
        agent_runs: dict[str, AgentRun] = runtime.state["agent_runs"]
        current_runs = _get_current_agent_runs(agent_runs)
        if not current_runs:
            logger.info("wait: tool_call_id=%s no active runs", runtime.tool_call_id)
            return NO_ACTIVE_RUNS_ERROR
        next_response_sequence_number = len(agent_runs) - len(current_runs)

        deadline = time.monotonic() + WAIT_TIMEOUT_SECONDS
        polls = 0
        while True:
            polls += 1
            tool_output: list[dict[str, Any]] = []
            updated_runs: dict[str, AgentRun] = {}

            for agent_run_id, agent_run in current_runs.items():
                agent = agents[agent_run["agent_id"]]
                client = clients.get_sync(agent["agent_type_id"])
                with trace("runs.get", metadata={
                    "agent_id": agent_run["agent_id"], "agent_run_id": agent_run["run_id"],
                }):
                    run = client.runs.get(
                        thread_id=agent["thread_id"],
                        run_id=agent_run["run_id"],
                    )

                status = _normalize_run_status(run["status"])
                logger.debug(
                    "wait: tool_call_id=%s run_id=%s poll=%d status=%s",
                    runtime.tool_call_id, agent_run_id, polls, status,
                )
                if status not in TERMINAL_STATUSES:
                    if status != agent_run["status"]:
                        updated_runs[agent_run_id] = {
                            **agent_run,
                            "status": status,
                        }
                    continue

                with trace("threads.get_history", metadata={
                    "agent_run_id": run["run_id"], "thread_id": run["thread_id"],
                }):
                    history = client.threads.get_history(
                        thread_id=run["thread_id"],
                        limit=HISTORY_LIMIT,
                        metadata={"run_id": run["run_id"]},
                    )
                with trace("wait.process_result", metadata={"agent_run_id": run["run_id"]}):
                    if history:
                        if history[-1]["metadata"]["source"] != "input":
                            raise ValueError(
                                "History is truncated; increase `HISTORY_LIMIT`."
                            )

                        before_messages = history[-1]["values"]["messages"]
                        after_messages = history[0]["values"]["messages"]
                        messages = after_messages[len(before_messages) :]
                    else:
                        messages = []

                    tool_calls = _extract_agent_tool_calls(messages)

                    response = None
                    error = None
                    if status == "responded" and messages:
                        last_message = messages[-1]
                        if (
                            last_message["type"] != "ai"
                            or last_message["tool_calls"]
                            or last_message["content"] is None
                        ):
                            raise ValueError(
                                "Expected a final AI message without tool calls and with non-null content "
                                f"for responded run `{run['run_id']}`."
                            )
                        response = last_message["content"]
                        if not isinstance(response, str):
                            response = json.dumps(response, ensure_ascii=False)
                    elif status == "error":
                        with trace("threads.get", metadata={
                            "agent_run_id": run["run_id"], "thread_id": run["thread_id"],
                        }):
                            thread = client.threads.get(thread_id=run["thread_id"])
                        error = thread.get("error")
                        error = json.dumps(error, ensure_ascii=False) if error else None
                    if status == "responded" and (response is None or not response.strip()):
                        status = "error"
                        response = None
                        error = EMPTY_AGENT_RESPONSE_ERROR
                    # response, _ = _truncate_text(response, AGENT_RESPONSE_MAX_TOKENS)

                    logger.info(
                        "Collected run: tool_call_id=%s agent_id=%s run_id=%s status=%s "
                        "history_entries=%d messages=%d tool_calls=%d",
                        runtime.tool_call_id, agent_run["agent_id"], agent_run_id, status,
                        len(history), len(messages), len(tool_calls),
                    )
                    response_sequence_number = next_response_sequence_number + len(tool_output)
                    tool_output.append({
                        "agent_id": agent_run["agent_id"],
                        "agent_run_id": agent_run_id,
                        "status": status,
                        "response": response,
                        "error": error,
                        "tool_calls_count": len(tool_calls),
                    })
                    updated_runs[agent_run_id] = {
                        **agent_run,
                        "status": status,
                        "tool_calls": tool_calls,
                        "messages": messages,
                        "response": response,
                        "error": error,
                        "response_sequence_number": response_sequence_number,
                    }

            if tool_output:
                logger.info(
                    "wait: tool_call_id=%s polls=%d active=%d collected=%d",
                    runtime.tool_call_id, polls, len(current_runs), len(tool_output),
                )
                collected_at = time.time()
                for result in tool_output:
                    updated_runs[result["agent_run_id"]]["collected_at"] = collected_at
                with trace("wait.build_response"):
                    return Command(
                        update={
                            "messages": [
                                ToolMessage(
                                    json.dumps(tool_output, ensure_ascii=False),
                                    tool_call_id=runtime.tool_call_id,
                                )
                            ],
                            "agent_runs": updated_runs,
                        }
                    )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.info(
                    "wait: tool_call_id=%s polls=%d active=%d timeout",
                    runtime.tool_call_id, polls, len(current_runs),
                )
                return Command(
                    update={
                        "messages": [
                            ToolMessage(
                                WAIT_TIMEOUT_ERROR,
                                tool_call_id=runtime.tool_call_id,
                            )
                        ],
                        "agent_runs": updated_runs,
                    }
                )

            with trace("wait.sleep"):
                time.sleep(min(WAIT_POLL_SECONDS, remaining))

    async def get_run(agent: Agent, agent_run: AgentRun) -> dict[str, Any]:
        client = clients.get_async(agent["agent_type_id"])
        with trace("runs.get", metadata={
            "agent_id": agent["agent_id"], "agent_run_id": agent_run["run_id"],
        }):
            return await client.runs.get(
                thread_id=agent["thread_id"], run_id=agent_run["run_id"],
            )

    async def await_(runtime: ToolRuntime) -> str | Command:
        agents: dict[str, Agent] = runtime.state["agents"]
        agent_runs: dict[str, AgentRun] = runtime.state["agent_runs"]
        current_runs = _get_current_agent_runs(agent_runs)
        if not current_runs:
            logger.info("wait: tool_call_id=%s no active runs", runtime.tool_call_id)
            return NO_ACTIVE_RUNS_ERROR
        next_response_sequence_number = len(agent_runs) - len(current_runs)

        deadline = asyncio.get_running_loop().time() + WAIT_TIMEOUT_SECONDS
        polls = 0
        while True:
            polls += 1
            tool_output: list[dict[str, Any]] = []
            updated_runs: dict[str, AgentRun] = {}

            with trace("wait.poll", metadata={"poll": polls, "run_count": len(current_runs)}):
                runs = await asyncio.gather(
                    *(get_run(agents[run["agent_id"]], run) for run in current_runs.values()),
                    return_exceptions=True,
                )
            for (agent_run_id, agent_run), run in zip(
                current_runs.items(), runs, strict=True
            ):
                if isinstance(run, (ReadError, RemoteProtocolError)):
                    logger.warning(
                        "wait: tool_call_id=%s run_id=%s poll=%d retrying after %s",
                        runtime.tool_call_id, agent_run_id, polls, type(run).__name__,
                    )
                    continue
                if isinstance(run, BaseException):
                    raise run

                status = _normalize_run_status(run["status"])
                logger.debug(
                    "wait: tool_call_id=%s run_id=%s poll=%d status=%s",
                    runtime.tool_call_id, agent_run_id, polls, status,
                )
                if status not in TERMINAL_STATUSES:
                    if status != agent_run["status"]:
                        updated_runs[agent_run_id] = {
                            **agent_run,
                            "status": status,
                        }
                    continue

                agent = agents[agent_run["agent_id"]]
                client = clients.get_async(agent["agent_type_id"])
                with trace("threads.get_history", metadata={
                    "agent_run_id": run["run_id"], "thread_id": run["thread_id"],
                }):
                    history = await client.threads.get_history(
                        thread_id=run["thread_id"],
                        limit=HISTORY_LIMIT,
                        metadata={"run_id": run["run_id"]},
                    )
                with trace("wait.process_result", metadata={"agent_run_id": run["run_id"]}):
                    if history:
                        if history[-1]["metadata"]["source"] != "input":
                            raise ValueError(
                                "History is truncated; increase `HISTORY_LIMIT`."
                            )

                        before_messages = history[-1]["values"]["messages"]
                        after_messages = history[0]["values"]["messages"]
                        messages = after_messages[len(before_messages) :]
                    else:
                        messages = []

                    tool_calls = _extract_agent_tool_calls(messages)

                    response = None
                    error = None
                    if status == "responded" and messages:
                        last_message = messages[-1]
                        if (
                            last_message["type"] != "ai"
                            or last_message["tool_calls"]
                            or last_message["content"] is None
                        ):
                            raise ValueError(
                                "Expected a final AI message without tool calls and with non-null content "
                                f"for responded run `{run['run_id']}`."
                            )
                        response = last_message["content"]
                        if not isinstance(response, str):
                            response = json.dumps(response, ensure_ascii=False)
                    elif status == "error":
                        with trace("threads.get", metadata={
                            "agent_run_id": run["run_id"], "thread_id": run["thread_id"],
                        }):
                            thread = await client.threads.get(thread_id=run["thread_id"])
                        error = thread.get("error")
                        error = json.dumps(error, ensure_ascii=False) if error else None
                    if status == "responded" and (response is None or not response.strip()):
                        status = "error"
                        response = None
                        error = EMPTY_AGENT_RESPONSE_ERROR
                    # response, _ = _truncate_text(response, AGENT_RESPONSE_MAX_TOKENS)

                    logger.info(
                        "Collected run: tool_call_id=%s agent_id=%s run_id=%s status=%s "
                        "history_entries=%d messages=%d tool_calls=%d",
                        runtime.tool_call_id, agent_run["agent_id"], agent_run_id, status,
                        len(history), len(messages), len(tool_calls),
                    )
                    response_sequence_number = next_response_sequence_number + len(tool_output)
                    tool_output.append({
                        "agent_id": agent_run["agent_id"],
                        "agent_run_id": agent_run_id,
                        "status": status,
                        "response": response,
                        "error": error,
                        "tool_calls_count": len(tool_calls),
                    })
                    updated_runs[agent_run_id] = {
                        **agent_run,
                        "status": status,
                        "tool_calls": tool_calls,
                        "messages": messages,
                        "response": response,
                        "error": error,
                        "response_sequence_number": response_sequence_number,
                    }

            if tool_output:
                logger.info(
                    "wait: tool_call_id=%s polls=%d active=%d collected=%d",
                    runtime.tool_call_id, polls, len(current_runs), len(tool_output),
                )
                collected_at = time.time()
                for result in tool_output:
                    updated_runs[result["agent_run_id"]]["collected_at"] = collected_at
                with trace("wait.build_response"):
                    return Command(
                        update={
                            "messages": [
                                ToolMessage(
                                    json.dumps(tool_output, ensure_ascii=False),
                                    tool_call_id=runtime.tool_call_id,
                                )
                            ],
                            "agent_runs": updated_runs,
                        }
                    )

            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                logger.info(
                    "wait: tool_call_id=%s polls=%d active=%d timeout",
                    runtime.tool_call_id, polls, len(current_runs),
                )
                return Command(
                    update={
                        "messages": [
                            ToolMessage(
                                WAIT_TIMEOUT_ERROR,
                                tool_call_id=runtime.tool_call_id,
                            )
                        ],
                        "agent_runs": updated_runs,
                    }
                )

            with trace("wait.sleep"):
                await asyncio.sleep(min(WAIT_POLL_SECONDS, remaining))

    return StructuredTool.from_function(
        func=wait,
        coroutine=await_,
        name="wait",
        description=WAIT_TOOL_DESCRIPTION.format(
            wait_timeout_seconds=WAIT_TIMEOUT_SECONDS,
            no_active_runs_error=NO_ACTIVE_RUNS_ERROR,
            wait_timeout_error=WAIT_TIMEOUT_ERROR,
            # agent_response_max_tokens=AGENT_RESPONSE_MAX_TOKENS,
        ),
    )


def _build_decomposer_agent_tools(
    agent_types: dict[str, AgentType],
    agent_recursion_limit: int | None,
    agent_run_budget_seconds: float | None = None,
    agent_shutdown_grace_seconds: float = 60.0,
) -> list[StructuredTool]:
    clients = _ClientCache(agent_types)
    return [
        _build_new_tool(agent_types, clients),
        _build_fork_tool(clients),
        _build_run_tool(
            clients, agent_recursion_limit,
            agent_run_budget_seconds, agent_shutdown_grace_seconds,
        ),
        _build_wait_tool(clients),
    ]


class DecomposerAgentMiddleware(AgentMiddleware[DecomposerAgentState, ContextT, ResponseT]):
    state_schema = DecomposerAgentState

    def __init__(
        self,
        agent_types: Sequence[AgentType],
        agent_recursion_limit: int | None,
        agent_run_budget_seconds: float | None = None,
        agent_shutdown_grace_seconds: float = 60.0,
    ) -> None:
        super().__init__()

        if not agent_types:
            msg = "At least one agent must be specified"
            raise ValueError(msg)

        ids = [a["agent_type_id"] for a in agent_types]
        agent_types = {a["agent_type_id"]: a for a in agent_types}
        if len(ids) > len(agent_types):
            seen = set()
            dupes = {id for id in ids if id in seen or seen.add(id)}
            raise ValueError(f"Duplicate agent type IDs: {dupes}")

        self.tools = _build_decomposer_agent_tools(
            agent_types, agent_recursion_limit,
            agent_run_budget_seconds, agent_shutdown_grace_seconds,
        )

    def before_agent(
        self,
        state: DecomposerAgentState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any]:
        run_id = runtime.execution_info.run_id or str(uuid4())
        prompt = next(
            message.text for message in reversed(state["messages"])
            if isinstance(message, HumanMessage)
        )
        run: AgentRun = {
            "agent_run_id": run_id,
            "agent_id": runtime.execution_info.thread_id or "decomposer",
            "run_id": run_id,
            "status": "running",
            "prompt": prompt,
            "started_at": time.time(),
        }
        return {
            "decomposer_agent_runs": [*state.get("decomposer_agent_runs", []), run],
            "run_prompt_counts": {},
            "shutdown_requested": False,
        }

    def after_agent(
        self,
        state: DecomposerAgentState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any]:
        runs = state["decomposer_agent_runs"]
        return {"decomposer_agent_runs": [
            *runs[:-1],
            {
                **runs[-1],
                "status": "responded",
                "response": state["messages"][-1].text,
                "collected_at": time.time(),
            },
        ]}

    def _prepare_model_request(self, request):
        if not request.state.get("shutdown_requested", False):
            return request
        if any(
            "response_sequence_number" not in run
            for run in request.state["agent_runs"].values()
        ):
            return request.override(
                tools=[tool for tool in self.tools if tool.name == "wait"],
                tool_choice="wait", response_format=None,
            )
        return request.override(tools=[], tool_choice=None, response_format=None)

    def wrap_model_call(self, request, handler):
        return handler(self._prepare_model_request(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._prepare_model_request(request))

    def after_model(
        self,
        state: DecomposerAgentState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        ai_message = next(
            (
                message
                for message in reversed(state["messages"])
                if isinstance(message, AIMessage)
            ),
            None,
        )
        if ai_message is None:
            raise RuntimeError("Expected the state to contain an AIMessage.")

        tool_calls = ai_message.tool_calls
        if state.get("shutdown_requested", False) and tool_calls:
            active_runs = any(
                "response_sequence_number" not in run
                for run in state["agent_runs"].values()
            )
            if not active_runs or any(call["name"] != "wait" for call in tool_calls):
                raise ValueError("Decomposer requested tools instead of completing graceful shutdown")
        agent_run_counts = Counter(
            tool_call["args"]["agent_id"]
            for tool_call in tool_calls
            if tool_call["name"] == "run"
            and isinstance(tool_call["args"].get("agent_id"), str)
        )
        prompt_counts = Counter(
            tool_call["args"]["prompt"]
            for tool_call in tool_calls
            if tool_call["name"] == "run"
            and isinstance(tool_call["args"].get("prompt"), str)
        )
        run_prompt_counts = Counter(state.get("run_prompt_counts", {}))
        run_prompt_counts.update(prompt_counts)
        if not state.get("shutdown_requested", False) and any(
            prompt.strip() and run_prompt_counts[prompt] >= 3 for prompt in prompt_counts
        ):
            return {
                "run_prompt_counts": dict(run_prompt_counts),
                "shutdown_requested": True,
                "messages": [
                    ToolMessage(
                        content=DECOMPOSER_GRACEFUL_SHUTDOWN_REQUEST,
                        tool_call_id=call["id"], name=call["name"],
                    )
                    for call in tool_calls
                ] + [HumanMessage(content=DECOMPOSER_GRACEFUL_SHUTDOWN_REQUEST)],
                "jump_to": "model",
            }
        forked_agent_ids = {
            tool_call["args"]["agent_id"]
            for tool_call in tool_calls
            if tool_call["name"] == "fork"
            and isinstance(tool_call["args"].get("agent_id"), str)
        }
        rejected_calls = []
        for tool_call in tool_calls:
            if tool_call["name"] == "wait" and len(tool_calls) > 1:
                error = PARALLEL_WAIT_CALL_ERROR
            elif (
                tool_call["name"] in {"run", "fork"}
                and isinstance(tool_call["args"].get("agent_id"), str)
                and tool_call["args"]["agent_id"] in forked_agent_ids
                and agent_run_counts[tool_call["args"]["agent_id"]] > 0
            ):
                error = PARALLEL_FORK_RUN_CALL_ERROR
            elif (
                tool_call["name"] == "run"
                and isinstance(tool_call["args"].get("agent_id"), str)
                and agent_run_counts[tool_call["args"]["agent_id"]] > 1
            ):
                error = PARALLEL_RUN_CALL_ERROR
            elif (
                tool_call["name"] == "run"
                and isinstance(tool_call["args"].get("prompt"), str)
                and prompt_counts[tool_call["args"]["prompt"]] > 1
            ):
                error = DUPLICATE_PROMPT_ERROR
            else:
                continue
            rejected_calls.append(
                ToolMessage(
                    content=error,
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                )
            )
        if rejected_calls:
            return {"messages": rejected_calls, "run_prompt_counts": dict(run_prompt_counts)}

        if tool_calls:
            return {"run_prompt_counts": dict(run_prompt_counts)}

        agent_runs = state["agent_runs"]
        if any("response_sequence_number" not in run for run in agent_runs.values()):
            error = EARLY_RESPONSE_ERROR
        elif not ai_message.text.strip():
            error = EMPTY_RESPONSE_ERROR
        else:
            return None

        return {
            "messages": [HumanMessage(content=error)],
            "jump_to": "model",
        }


def create_decomposer_agent(
    *,
    decomposer_model: str | BaseChatModel,
    agent_types: Sequence[AgentType],
    middleware: Sequence[AgentMiddleware] | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    decomposer_recursion_limit: int | None = None,
    agent_recursion_limit: int | None = None,
    agent_run_budget_seconds: float | None = 60.0,
    agent_shutdown_grace_seconds: float = 60.0,
) -> CompiledStateGraph:
    if agent_run_budget_seconds is not None and agent_run_budget_seconds <= 0:
        raise ValueError("agent_run_budget_seconds must be positive")
    if agent_shutdown_grace_seconds <= 0:
        raise ValueError("agent_shutdown_grace_seconds must be positive")
    decomposer_middelware = DecomposerAgentMiddleware(
        agent_types, agent_recursion_limit,
        agent_run_budget_seconds, agent_shutdown_grace_seconds,
    )

    run_budget_convention = ""
    if agent_run_budget_seconds is not None:
        run_budget_convention = RUN_BUDGET_CONVENTION.format(
            agent_run_budget_seconds=agent_run_budget_seconds,
            agent_shutdown_grace_seconds=agent_shutdown_grace_seconds,
        ) + "\n\n"
    system_prompt = DECOMPOSER_SYSTEM_PROMPT.replace(
        "{run_budget_convention}\n\n", run_budget_convention,
    )

    agent = create_agent(
        model=decomposer_model,
        tools=[],
        system_prompt=system_prompt,
        middleware=[
            decomposer_middelware,
            *(middleware or []),
        ],
        context_schema=context_schema,
        checkpointer=checkpointer,
    )
    if decomposer_recursion_limit is not None:
        agent = agent.with_config(recursion_limit=decomposer_recursion_limit)
    return agent
