"""Compact tool activity and orchestrator callbacks for the chat interface."""

import re

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage, ToolMessage
from prompt_toolkit.utils import get_cwidth


TOOL_ICONS = {"new": "✦", "fork": "⑂", "run": "▶", "wait": "◷"}


class ResponseProgress:
    def __init__(self, index, changed):
        self.index = index
        self.changed = changed
        self.finished = False
        self.stage = "Preparing request"
        self.events = []
        self.requests = {}
        self.tools = {}
        self.agents = set()
        self.runs = {}
        self.run_calls = {}
        self.model_requests = 0

    def record(self, text, stage=None):
        self.events.append({"tool": None, "text": text, "details": ""})
        self.set_stage(stage or self.stage)

    def set_stage(self, stage):
        self.stage = stage
        self.changed()

    def finish(self, stage):
        self.finished = True
        self.record(stage, stage)

    @staticmethod
    def preview(text):
        text = " ".join(str(text).split())
        return text if len(text) <= 240 else text[:237] + "…"

    def requested(self, call):
        if call["id"] in self.requests:
            return
        args = call["args"]
        agent = args.get("agent_id")
        event = {
            "tool": call["name"],
            "text": (f"agent {agent[:8]} · " if agent else "") + "queued",
            "details": "Prompt: " + self.preview(args["prompt"]) if args.get("prompt") else "",
            "status": "queued",
        }
        if agent:
            self.agents.add(agent)
        self.requests[call["id"]] = event
        self.events.append(event)
        self.set_stage("Dispatching tools")

    def state_update(self, update):
        calls = [
            self.requests[message.tool_call_id]
            for message in update.get("messages", [])
            if isinstance(message, ToolMessage) and message.tool_call_id in self.requests
        ]
        event = calls[0] if len(calls) == 1 else None
        for agent_id, agent in update.get("agents", {}).items():
            if agent_id in self.agents:
                continue
            self.agents.add(agent_id)
            source = agent.get("forked_from")
            text = f"agent {agent_id[:8]} · from agent {source[:8]}" if source else f"agent {agent_id[:8]} · created"
            if event:
                event["text"] = text
            else:
                self.events.append({"tool": "fork" if source else "new", "text": text, "details": ""})
        collected = []
        for run_id, run in update.get("agent_runs", {}).items():
            self.agents.add(run["agent_id"])
            marker = (run["status"], "response_sequence_number" in run)
            if self.runs.get(run_id) == marker:
                continue
            self.runs[run_id] = marker
            if event and event["tool"] == "run":
                self.run_calls[run_id] = event
                event["text"] = f"agent {run['agent_id'][:8]} · running"
            if marker[1]:
                run_event = self.run_calls.get(run_id)
                if run_event:
                    label = "report ready" if run["status"] == "responded" else run["status"]
                    run_event["text"] = f"agent {run['agent_id'][:8]} · {label}"
                report = run.get("response") or run.get("error") or run["status"]
                collected.append(f"agent {run['agent_id'][:8]}: {self.preview(report)}")
        if collected and event and event["tool"] == "wait":
            count = len(collected)
            event["text"] = f"collected {count} report{'s' if count != 1 else ''}"
            event["details"] = "\n".join(collected)
        self.changed()

    def render(self, expanded, width=80):
        events = list(self.events)
        if not self.finished and self.stage in ("Waiting for model", "Preparing answer"):
            events.append({"tool": None, "text": self.stage, "details": ""})
        if not expanded:
            events = events[-3:]
        fragments = [("class:muted", "  Activity · Ctrl+T " + ("collapse" if expanded else "expand") + "\n")]
        ids = {agent[:8] for agent in self.agents}
        pattern = re.compile(r"agent (?:" + "|".join(re.escape(value) for value in ids) + r")\b") if ids else None

        def text_fragments(text, style, prefix_width):
            text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
            # Keep activity rows to one terminal line, including wide glyphs.
            while text and get_cwidth(text) > max(1, width - prefix_width):
                text = text[:-2] + "…"
            start = 0
            if pattern:
                for match in pattern.finditer(text):
                    fragments.append((style, text[start:match.start()]))
                    fragments.append(("class:agent", match[0]))
                    start = match.end()
            fragments.append((style, text[start:] + "\n"))

        for event in events:
            tool = event["tool"]
            icon = TOOL_ICONS.get(tool, "·") if tool else {"Responded": "✓", "Interrupted": "■", "Failed": "!"}.get(event["text"], "·")
            fragments.append(("class:muted", "    " + icon + " "))
            if tool:
                fragments.append(("class:tool", tool + "  "))
            text_fragments(event["text"], "class:muted", get_cwidth("    " + icon + " ") + (len(tool) + 2 if tool else 0))
            if expanded and event["details"]:
                for line in event["details"].splitlines():
                    fragments.append(("class:muted", "      ↳ "))
                    text_fragments(line, "class:muted", 8)
        fragments.append(("", "\n"))
        return fragments


class ProgressCallbacks(AsyncCallbackHandler):
    def __init__(self, progress, model):
        self.progress = progress
        self.model = model

    async def on_chat_model_start(self, serialized, messages, **kwargs):
        if "moska:compaction" in kwargs.get("tags", []):
            return
        self.progress.model_requests += 1
        self.progress.set_stage("Waiting for model")

    async def on_custom_event(self, name, data, **kwargs):
        if name == "moska.compaction":
            if data["stage"] == "start":
                self.progress.record("Compacting earlier context", "Compacting context")
            else:
                self.progress.record("Context compacted · full history retained", "Continuing orchestration")

    async def on_llm_end(self, response, **kwargs):
        if "moska:compaction" in kwargs.get("tags", []):
            return
        for generations in response.generations:
            for generation in generations:
                message = generation.message
                for call in message.tool_calls:
                    self.progress.requested(call)
                if message.text and not message.tool_calls:
                    self.progress.set_stage("Preparing answer")

    async def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        name = serialized.get("name", "tool")
        event = next((event for event in self.progress.events
                      if event["tool"] == name and event.get("status") == "queued"), None)
        self.progress.tools[run_id] = event
        if event:
            event["status"] = "running"
            event["text"] = event["text"].removesuffix("queued") + "running"
        self.progress.set_stage("Waiting for subagents" if name == "wait" else f"Calling {name}")

    async def on_tool_end(self, output, *, run_id, **kwargs):
        update = getattr(output, "update", None)
        if isinstance(update, dict):
            self.progress.state_update(update)
        event = self.progress.tools.get(run_id)
        if event:
            event["status"] = "done"
            if event["tool"] != "run" and event["text"].endswith("running"):
                event["text"] = event["text"].removesuffix("running") + "done"
        self.progress.set_stage("Continuing orchestration")

    async def on_tool_error(self, error, *, run_id, **kwargs):
        event = self.progress.tools.get(run_id)
        if event:
            event["status"] = "failed"
            event["text"] = "failed"
        self.progress.set_stage("Handling tool error")


