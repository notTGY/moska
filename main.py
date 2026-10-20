"""Terminal chat for the decomposer with a mock worker transport."""

import asyncio
import json
import re
import webbrowser
from copy import deepcopy
from unittest.mock import patch
from uuid import uuid4

import httpx
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_sdk.client import LangGraphClient
from prompt_toolkit import Application
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.data_structures import Point
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.filters import Condition, has_focus
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import (
    Float, FloatContainer, FormattedTextControl, HSplit, Layout, VSplit, Window,
)
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import Dialog, Label, RadioList, TextArea

from decomposer import create_decomposer_agent
from adapters.openai import ConnectionError, OpenAIConnection
from adapters.openai_model import ChatGPTModel


# prompt-toolkit has no Shift+Enter key name. Decode terminal encodings as
# Control-J (newline), keeping ordinary Enter as Control-M (send).
ANSI_SEQUENCES["\x1b[13;2u"] = Keys.ControlJ
ANSI_SEQUENCES["\x1b[27;2;13~"] = Keys.ControlJ
ANSI_SEQUENCES["\x1b[13u"] = Keys.ControlM
ANSI_SEQUENCES["\x1b[27u"] = Keys.Escape
ANSI_SEQUENCES["\x1b[127u"] = Keys.ControlH
for letter in "abcdefghijklmnopqrstuvwxyz":
    ANSI_SEQUENCES[f"\x1b[{ord(letter)};5u"] = Keys(f"c-{letter}")


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
        self.progress.model_requests += 1
        self.progress.set_stage("Waiting for model")

    async def on_llm_end(self, response, **kwargs):
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


COMMANDS = {
    "/connect": "Continue with ChatGPT",
    "/model": "Choose model and reasoning effort",
    "/help": "Show commands and demo information",
    "/clear": "Clear the visible conversation",
    "/exit": "Exit moska",
}


class SlashCompleter(Completer):
    def get_completions(self, document, complete_event):
        prefix = document.text_before_cursor
        if prefix.startswith("/") and not any(c.isspace() for c in prefix):
            for command, description in COMMANDS.items():
                if command.startswith(prefix):
                    yield Completion(command, start_position=-len(prefix), display_meta=description)


class ChatUI:
    def __init__(self, agent, *, agent_factory=None, **app_options):
        self.agent = agent
        self.agent_factory = agent_factory
        self.connection = OpenAIConnection()
        self.account = None
        self.model = None
        self.reasoning_effort = None
        self.picker = None
        self.pick_future = None
        self.config = {"configurable": {"thread_id": str(uuid4())}}
        self.fragments = []
        self.progress = []
        self.expanded_tools = False
        self.follow_output = True
        self.cursor_line = 0
        self.task = None
        self.activity = "interface demo · mock workers"
        self.composer = TextArea(
            prompt=" › ", multiline=True, height=Dimension(min=1, max=6),
            dont_extend_height=True,
            style="class:composer", completer=SlashCompleter(),
            complete_while_typing=True, history=InMemoryHistory(),
        )
        self.transcript = FormattedTextControl(
            self.render_transcript,
            get_cursor_position=lambda: Point(x=0, y=self.cursor_line),
            show_cursor=False,
        )
        keys = KeyBindings()

        @keys.add("enter", filter=has_focus(self.composer))
        def send(event):
            buffer = self.composer.buffer
            if buffer.complete_state and buffer.complete_state.current_completion:
                buffer.complete_state = None
                return
            buffer.cancel_completion()
            text = buffer.text.strip()
            if not text or (self.task and not self.task.done() and text != "/exit"):
                return
            buffer.append_to_history()
            buffer.reset()
            if text == "/exit":
                event.app.exit()
            elif text == "/clear":
                self.fragments.clear()
                self.progress.clear()
                self.follow_output = True
                self.cursor_line = 0
            elif text == "/connect" or text.startswith("/connect "):
                self.task = event.app.create_background_task(self.connect(text.removeprefix("/connect").strip()))
            elif text == "/model" or text.startswith("/model "):
                self.task = event.app.create_background_task(self.choose_model(text.removeprefix("/model").strip()))
            elif text == "/help":
                self.add("moska", "Commands\n" + "\n".join(
                    f"{command}  {description}" for command, description in COMMANDS.items()
                ) + "\n\nEach message runs a mock worker that reports hello.\n"
                    "Use /model to replace the demo coordinator with your selected model.\n"
                    "Ctrl+T shows tool activity. Esc interrupts a response; /exit works while busy.\n"
                    "/connect opens ChatGPT sign-in; /connect new adds an account.\n"
                    "/connect accounts lists saved accounts; /connect 1 selects one.\n"
                    "https://python-prompt-toolkit.readthedocs.io/")
            elif text.startswith("/"):
                self.add("system", "Unknown command. Type / to see available commands.")
            else:
                self.follow_output = True
                self.add("you", text)
                self.task = event.app.create_background_task(self.respond(text))

        @keys.add("escape", filter=Condition(lambda: self.picker is not None))
        def dismiss(event):
            if not self.pick_future.done():
                self.pick_future.set_result(None)

        @keys.add("escape", filter=Condition(
            lambda: self.picker is None and self.task is not None and not self.task.done()
        ))
        def interrupt(event):
            self.task.cancel()

        @keys.add("c-t")
        def toggle_tools(event):
            self.expanded_tools = not self.expanded_tools
            self.refresh_progress()

        @keys.add("c-j", filter=has_focus(self.composer))
        def newline(event):
            self.composer.buffer.insert_text("\n")

        @keys.add("c-d")
        def quit_app(event):
            event.app.exit()

        @keys.add("c-c")
        def cancel(event):
            if self.task and not self.task.done():
                self.task.cancel()
            else:
                self.composer.buffer.reset()

        @keys.add("pageup")
        def scroll_up(event):
            self.follow_output = False
            self.cursor_line = max(0, self.cursor_line - 10)

        @keys.add("pagedown")
        def scroll_down(event):
            self.cursor_line = min(self.last_line, self.cursor_line + 10)
            self.follow_output = self.cursor_line == self.last_line

        body = HSplit([
            Window(FormattedTextControl([
                ("class:brand", "  moska"),
                ("class:muted", "  /  local agent workspace"),
            ]), height=2),
            VSplit([Window(width=2), Window(self.transcript, wrap_lines=True), Window(width=2)]),
            Window(height=1),
            CompletionsMenu(max_height=5),
            HSplit([
                Window(height=1, style="class:composer"),
                self.composer,
                Window(height=1, style="class:composer"),
            ], style="class:composer"),
            Window(FormattedTextControl(lambda: [
                ("class:muted", "  Enter send · Shift+Enter newline · Ctrl+T details · Esc interrupt"),
            ]), height=1),
            Window(FormattedTextControl(lambda: [
                ("class:accent", "  " + self.activity),
            ]), height=1),
        ])
        self.root = FloatContainer(body, floats=[])
        self.app = Application(
            layout=Layout(self.root, focused_element=self.composer),
            key_bindings=keys, full_screen=True, mouse_support=True,
            style=Style.from_dict({
                "": "bg:default fg:default",
                "brand": "#a6d8cc bold",
                "accent": "#a6d8cc",
                "muted": "#888888",
                "tool": "bold",
                "agent": "#a6d8cc bold",
                "you": "#eeeeee bold",
                "moska": "#a6d8cc bold",
                "system": "#d8b98c",
                "link": "#9dbfe5 underline",
                "composer": "bg:#292929 #eeeeee",
                "completion-menu": "bg:#292929 #dddddd",
                "completion-menu.completion.current": "bg:#444444 #ffffff",
                "completion-menu.meta.completion": "bg:#292929 #999999",
                "completion-menu.meta.completion.current": "bg:#444444 #dddddd",
                "dialog.body": "bg:#242424 #dddddd",
                "dialog frame.label": "#a6d8cc bold",
                "radio-selected": "bg:#444444 #ffffff",
                "radio-checked": "#a6d8cc",
            }),
            **app_options,
        )
        self.add("moska", "What would you like to work on?\n"
                 "Mock workers always report hello. Use /model to choose the orchestrator.")
        try:
            self.account = self.connection.current_account()
            self.activate_model()
            self.activity = self.idle_activity
        except ConnectionError as error:
            self.add("system", str(error))

    @property
    def idle_activity(self):
        if self.model:
            return f"{self.model} · {self.reasoning_effort or 'default'} reasoning · mock workers"
        if self.account and self.account.plan_enabled:
            return "ChatGPT connected · interface demo · mock workers"
        return "interface demo · mock workers"

    def activate_model(self):
        self.model = self.connection.selected_model() if self.account and self.account.plan_enabled else None
        self.reasoning_effort = self.connection.selected_reasoning_effort() if self.model else None
        if self.agent_factory:
            model = ChatGPTModel(
                connection=self.connection, model_name=self.model, reasoning_effort=self.reasoning_effort,
            ) if self.model else DemoCoordinator()
            self.agent = self.agent_factory(model)

    async def pick(self, title, values, default=None):
        self.pick_future = asyncio.get_running_loop().create_future()
        self.picker = RadioList(values=values, default=default, select_on_focus=True)

        @self.picker.control.key_bindings.add("enter")
        def accept(event):
            if not self.pick_future.done():
                self.pick_future.set_result(self.picker.current_value)

        dialog = Float(content=Dialog(
            title=title, width=78,
            body=HSplit([
                Label("  ↑↓ choose · Enter select · Esc cancel\n"),
                self.picker,
            ]),
        ))
        self.root.floats.append(dialog)
        self.app.layout.focus(self.picker)
        self.app.invalidate()
        try:
            return await self.pick_future
        finally:
            self.root.floats.remove(dialog)
            self.picker = None
            self.pick_future = None
            self.app.layout.focus(self.composer)

    async def choose_model(self, option=""):
        self.activity = "loading models… · Ctrl+C to cancel"
        self.app.invalidate()
        try:
            args = option.split()
            if len(args) > 2:
                raise ConnectionError("Use /model or /model <model> <effort>.")
            slug = args[0] if args else None
            effort = args[1] if len(args) == 2 else None
            models = await self.connection.list_models()
            if not models:
                raise ConnectionError("No selectable models are available for this account.")
            if not slug:
                slug = await self.pick(
                    "Orchestrator model",
                    [(model["slug"], model["display_name"]) for model in models],
                    self.model,
                )
                if slug is None:
                    return
            model = next((model for model in models if model["slug"] == slug), None)
            if model is None:
                raise ConnectionError("That model is not available. Use /model to choose one.")
            levels = model.get("supported_reasoning_levels", [])
            if levels and effort is None:
                effort = await self.pick(
                    "Reasoning effort · " + model["display_name"],
                    [(level["effort"], level["effort"] + (
                        " — " + level["description"] if level.get("description") else ""
                    )) for level in levels],
                    (self.reasoning_effort if slug == self.model else None) or model.get("default_reasoning_level"),
                )
                if effort is None:
                    return
            await self.connection.select_model(slug, reasoning_effort=effort)
            self.activate_model()
            self.add("system", f"Orchestrator model: {self.model} · {self.reasoning_effort or 'default'} reasoning.\n"
                     "Subagents still use mock workers.")
        except asyncio.CancelledError:
            raise
        except ConnectionError as error:
            self.add("system", str(error))
        finally:
            self.activity = self.idle_activity
            self.app.invalidate()

    async def connect(self, option=""):
        self.activity = "waiting for ChatGPT sign-in · Ctrl+C to cancel"
        self.app.invalidate()
        try:
            accounts = self.connection.accounts()
            if option == "accounts":
                lines = [
                    f"{index}. {account.email or 'ChatGPT account'}"
                    + (" · active" if self.account and self.account.client_id == account.client_id else "")
                    for index, account in enumerate(accounts, 1)
                ]
                self.add("system", "\n".join(lines) + "\nUse /connect <number> or /connect new."
                         if lines else "No saved accounts. Use /connect to sign in.")
                return
            client_id = None
            if option.isdecimal() and 1 <= int(option) <= len(accounts):
                client_id = accounts[int(option) - 1].client_id
            elif option not in ("", "openai", "new"):
                self.add("system", "Use /connect, /connect new, or /connect accounts.")
                return

            def notify(url):
                self.add("system", "Continue with ChatGPT\n"
                         "Sign in and allow Moska to use your ChatGPT plan.\n"
                         "If the browser doesn't open, click this link:\n" + url)

            previous = self.account
            self.account = await self.connection.sign_in(
                notify, new_account=option == "new", client_id=client_id,
            )
            if not previous or previous.client_id != self.account.client_id:
                self.config = {"configurable": {"thread_id": str(uuid4())}}
            self.activate_model()
            if self.account.plan_enabled:
                self.add("system", f"Connected to ChatGPT as {self.account.email or 'your account'}.\n"
                         "ChatGPT plan usage is enabled. Use /model to choose the orchestrator model.")
            else:
                self.add("system", "Signed in, but ChatGPT plan usage wasn't granted.\n"
                         "Run /connect again and allow plan usage to enable model access.")
        except asyncio.CancelledError:
            self.add("system", "Sign-in cancelled. Your previous connection is unchanged.")
            raise
        except ConnectionError as error:
            self.add("system", str(error))
        finally:
            self.activity = self.idle_activity
            self.app.invalidate()

    def render_transcript(self):
        progress = {item.index: item for item in self.progress}
        fragments = []
        width = self.app.output.get_size().columns - 4
        for index, fragment in enumerate(self.fragments):
            if index in progress:
                fragments.extend(progress[index].render(self.expanded_tools, width))
            else:
                fragments.append(fragment)
        return fragments

    def refresh_progress(self):
        if self.follow_output:
            self.cursor_line = self.last_line
        self.app.invalidate()

    @property
    def last_line(self):
        return max(0, sum(fragment[1].count("\n") for fragment in self.render_transcript()) - 1)

    def add(self, role, text):
        text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
        self.fragments.append((f"class:{role}", role + "\n"))
        start = 0
        for match in re.finditer(r"https?://[^\s<>]+", text):
            self.fragments.append(("", text[start:match.start()]))
            url = match[0]

            def open_link(event, url=url):
                if event.event_type == MouseEventType.MOUSE_UP:
                    webbrowser.open(url)

            self.fragments.append(("class:link", url, open_link))
            start = match.end()
        self.fragments.append(("", text[start:] + "\n\n"))
        if self.follow_output:
            self.cursor_line = self.last_line
        self.app.invalidate()

    async def respond(self, prompt):
        progress = ResponseProgress(len(self.fragments), self.refresh_progress)
        self.fragments.append(("", ""))
        self.progress.append(progress)
        progress.record("Message received", "Preparing request")
        self.activity = "working… · Esc interrupt · /exit quit"
        callbacks = ProgressCallbacks(progress, self.model or "scripted demo")
        try:
            async for update in self.agent.astream(
                {"messages": [HumanMessage(content=prompt)], "agents": {}, "agent_runs": {}},
                config={**self.config, "callbacks": [callbacks]}, stream_mode="updates",
            ):
                for value in update.values():
                    if not isinstance(value, dict):
                        continue
                    progress.state_update(value)
                    for message in value.get("messages", []):
                        if isinstance(message, AIMessage) and message.content and not message.tool_calls:
                            self.add("moska", message.text)
                self.app.invalidate()
            progress.finish("Responded")
        except asyncio.CancelledError:
            progress.finish("Interrupted")
            self.add("system", "Interrupted.")
            self.config = {"configurable": {"thread_id": str(uuid4())}}
            raise
        except Exception as error:
            progress.finish("Failed")
            self.add("system", f"Error: {error}")
            self.config = {"configurable": {"thread_id": str(uuid4())}}
        finally:
            self.activity = self.idle_activity
            self.app.invalidate()


async def chat():
    workers = MockWorkers()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(workers.handle), base_url="http://mock.local",
    ) as http:
        client = LangGraphClient(http)
        # Redirect only the SDK factory; decomposer tools and prompts stay intact.
        with patch("decomposer.core.get_client", return_value=client):
            memory = InMemorySaver()

            def build_agent(model):
                return create_decomposer_agent(
                    decomposer_model=model,
                    agent_types=[{
                        "agent_type_id": "mock", "description": 'Does nothing and reports "hello".',
                        "assistant_id": "mock", "url": "http://mock.local",
                    }],
                    checkpointer=memory, agent_run_budget_seconds=None,
                )

            ui = ChatUI(build_agent(DemoCoordinator()), agent_factory=build_agent)
            # Ask supporting terminals to distinguish modified Enter. Restore
            # their previous keyboard mode even when the application fails.
            try:
                ui.app.output.write_raw("\x1b[>1u")
                ui.app.output.flush()
                await ui.app.run_async()
            finally:
                ui.app.output.write_raw("\x1b[<u")
                ui.app.output.flush()


def main():
    try:
        asyncio.run(chat())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
