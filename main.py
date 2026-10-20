"""Small chat UI using the real decomposer with an entirely local demo backend."""

import asyncio
import json
import re
from copy import deepcopy
from unittest.mock import patch
from uuid import uuid4

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_sdk.client import LangGraphClient
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout

from decomposer import create_decomposer_agent


class DemoCoordinator(BaseChatModel):
    """Script the tool cycle so the UI demo needs no model or credentials."""

    @property
    def _llm_type(self):
        return "moska-interface-demo"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        last = messages[-1]
        if isinstance(last, HumanMessage):
            name, args = "new", {"agent_type_id": "mock"}
        elif isinstance(last, ToolMessage):
            result = json.loads(last.content)
            if last.name == "new":
                prompt = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
                name, args = "run", {"agent_id": result["agent_id"], "prompt": prompt}
            elif last.name == "run":
                name, args = "wait", {}
            else:
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=result[0]["response"]))])
        else:
            raise RuntimeError("Unexpected demo conversation state")
        message = AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": str(uuid4())}])
        return ChatResult(generations=[ChatGeneration(message=message)])


class MockWorkers:
    """In-memory LangGraph transport; workers do no work and report hello."""

    def __init__(self):
        self.threads = {}
        self.runs = {}

    def handle(self, request):
        parts = request.url.path.strip("/").split("/")
        body = json.loads(request.content) if request.content else {}
        if parts == ["threads"] and request.method == "POST":
            thread_id = str(uuid4())
            self.threads[thread_id] = []
            return httpx.Response(200, json={"thread_id": thread_id})

        thread_id = parts[1]
        if parts[2:] == ["state"]:
            return httpx.Response(200, json={"checkpoint": {"checkpoint_id": thread_id} if self.threads[thread_id] else None})
        if parts[2:] == ["copy"]:
            copy_id = str(uuid4())
            self.threads[copy_id] = deepcopy(self.threads[thread_id])
            return httpx.Response(200, json={"thread_id": copy_id})
        if parts[2:] == ["runs"] and request.method == "POST":
            run_id = str(uuid4())
            before = self.threads[thread_id] + body["input"]["messages"]
            after = before + [{"type": "ai", "content": "hello", "tool_calls": []}]
            self.threads[thread_id] = after
            self.runs[run_id] = {
                "run_id": run_id, "thread_id": thread_id, "status": "success",
                "history": [
                    {"values": {"messages": after}, "metadata": {"source": "loop"}},
                    {"values": {"messages": before}, "metadata": {"source": "input"}},
                ],
            }
            return httpx.Response(200, json={"run_id": run_id, "status": "pending"})
        if parts[2:3] == ["runs"] and request.method == "GET":
            run = self.runs[parts[3]]
            return httpx.Response(200, json={k: v for k, v in run.items() if k != "history"})
        if parts[2:] == ["history"]:
            return httpx.Response(200, json=self.runs[body["metadata"]["run_id"]]["history"])
        return httpx.Response(404, json={"error": "Unknown mock endpoint"})


def write(text):
    """Use native terminal hyperlinks (OSC 8) for HTTP(S) URLs."""
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    text = re.sub(r"https?://[^\s<>]+", lambda m: f"\033]8;;{m[0]}\033\\{m[0]}\033]8;;\033\\", text)
    print(text, flush=True)


async def chat():
    keys = KeyBindings()

    @keys.add("escape", "enter")
    def newline(event):
        event.current_buffer.insert_text("\n")

    session = PromptSession(key_bindings=keys)
    workers = MockWorkers()
    async with httpx.AsyncClient(transport=httpx.MockTransport(workers.handle), base_url="http://mock.local") as http:
        client = LangGraphClient(http)
        # Redirect only the SDK factory; decomposer tools and prompts stay intact.
        with patch("decomposer.core.get_client", return_value=client), patch_stdout(raw=True):
            agent = create_decomposer_agent(
                decomposer_model=DemoCoordinator(),
                agent_types=[{"agent_type_id": "mock", "description": 'Does nothing and reports "hello".', "assistant_id": "mock", "url": "http://mock.local"}],
                checkpointer=InMemorySaver(),
                agent_run_budget_seconds=None,
            )
            config = {"configurable": {"thread_id": str(uuid4())}}
            write("moska · interface demo\nEnter to send · Alt+Enter for newline · Ctrl+D or /quit to exit\n/help for help\n")
            while True:
                try:
                    prompt = (await session.prompt_async("› ")).strip()
                except EOFError:
                    break
                except KeyboardInterrupt:
                    continue
                if not prompt:
                    continue
                if prompt == "/quit":
                    break
                if prompt == "/help":
                    write("\nDemo: each message creates a mock worker, runs it, and collects hello.\nThe coordinator is scripted; no model or Codex is called.\nLinks: https://python-prompt-toolkit.readthedocs.io/\n")
                    continue
                try:
                    async for update in agent.astream(
                        {"messages": [HumanMessage(content=prompt)], "agents": {}, "agent_runs": {}},
                        config=config, stream_mode="updates",
                    ):
                        for value in update.values():
                            if not isinstance(value, dict):
                                continue
                            for message in value.get("messages", []):
                                if isinstance(message, AIMessage):
                                    for call in message.tool_calls:
                                        write(f"  · {call['name']}")
                                    if message.content and not message.tool_calls:
                                        write(f"\nmoska  {message.text}\n")
                except Exception as error:
                    write(f"\nError: {error}\n")
                    # Start a fresh conversation after an incomplete tool cycle.
                    config = {"configurable": {"thread_id": str(uuid4())}}


def main():
    try:
        asyncio.run(chat())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
