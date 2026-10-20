"""Credential-free interface demo; never used for real worker execution."""

import json
from copy import deepcopy
from uuid import uuid4

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult


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


