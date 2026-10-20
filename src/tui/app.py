"""Full-screen terminal chat and configuration dialogs."""

import asyncio
import re
from uuid import uuid4

from prompt_toolkit import Application
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.data_structures import Point
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.filters import Condition, has_focus
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import (
    FloatContainer, FormattedTextControl, HSplit, Layout, VSplit, Window,
)
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Label, TextArea

from adapters.providers import MODEL_PROVIDERS, ProviderError as ConnectionError
from adapters.agents import AgentSettings
from tui.configuration import ConfigurationDialogs
from tui.transcript import SelectableTranscript, copy_text
from tui.progress import ResponseProgress
from tui.conversation import Conversation
from tui.markdown import markdown_fragments, plain_fragments
from tui.jobs import WorkerCommands


# prompt-toolkit has no Shift+Enter key name. Decode terminal encodings as
# Control-J (newline), keeping ordinary Enter as Control-M (send).
ANSI_SEQUENCES["\x1b[13;2u"] = Keys.ControlJ
ANSI_SEQUENCES["\x1b[27;2;13~"] = Keys.ControlJ
ANSI_SEQUENCES["\x1b[13u"] = Keys.ControlM
ANSI_SEQUENCES["\x1b[27u"] = Keys.Escape
ANSI_SEQUENCES["\x1b[127u"] = Keys.ControlH
for letter in "abcdefghijklmnopqrstuvwxyz":
    ANSI_SEQUENCES[f"\x1b[{ord(letter)};5u"] = Keys(f"c-{letter}")


COMMANDS = {
    "/agents": "Configure on-device agents",
    "/ps": "List agents, workspaces, prompts and runtime",
    "/stop": "Pause orchestration and stop all background agents",
    "/connect": "Connect an orchestrator model provider",
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


class ChatUI(ConfigurationDialogs, Conversation, WorkerCommands):
    def __init__(self, agent, *, agent_factory=None, worker_runtime=None,
                 agent_settings=None, session=None, prepare_turn=None, **app_options):
        self.agent = agent
        self.agent_factory = agent_factory
        self.worker_runtime = worker_runtime
        self.session = session
        self.prepare_turn = prepare_turn
        self._restoring = False
        self.provider = next(iter(MODEL_PROVIDERS.values()))
        self.connection = self.provider.create_connection()
        self.agent_settings = agent_settings or AgentSettings()
        self.account = None
        self.model = None
        self.reasoning_effort = None
        self.demo_mode = session.meta('demo_mode', True) if session else True
        self.picker = None
        self.pick_future = None
        self.config = {"configurable": {"thread_id": session.meta('thread_id', session.session_id) if session else str(uuid4())}}
        self.fragments = []
        self.progress = []
        self.expanded_tools = True
        self.follow_output = True
        self.cursor_line = 0
        self.task = None
        self.stop_task = None
        self.pending_messages = session.meta('pending_messages', []) if session else []
        self.responding = False
        self.restart_response = False
        self.preparing_turn = False
        self.activity = "interface demo · mock workers"
        self.composer = TextArea(
            prompt=" › ", multiline=True, height=Dimension(min=1, max=6),
            dont_extend_height=True,
            style="class:composer", completer=SlashCompleter(),
            complete_while_typing=True, history=InMemoryHistory(),
        )
        self.transcript = SelectableTranscript(
            self.render_transcript, lambda: self.app.invalidate(), self.copy_selection,
            scroll=self.scroll_transcript,
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
            if not text:
                return
            busy = self.task and not self.task.done()
            if self.stop_task and not self.stop_task.done() and text not in ('/exit', '/ps', '/help'):
                return
            if busy and text not in ('/exit', '/ps', '/stop', '/help') and (text.startswith('/') or not self.responding):
                return
            buffer.append_to_history()
            buffer.reset()
            if text == "/exit":
                event.app.exit()
            elif text == "/ps":
                self.show_processes()
            elif text == "/stop":
                self.stop_task = event.app.create_background_task(self.stop_workers())
            elif text == "/clear":
                if self.session:
                    self.session.append('clear', {})
                self.fragments.clear()
                self.progress.clear()
                self.follow_output = True
                self.cursor_line = 0
            elif text == "/connect" or text.startswith("/connect "):
                self.task = event.app.create_background_task(self.connect(text.removeprefix("/connect").strip()))
            elif text == "/model" or text.startswith("/model "):
                self.task = event.app.create_background_task(self.choose_model(text.removeprefix("/model").strip()))
            elif text == "/agents":
                self.task = event.app.create_background_task(self.configure_agents())
            elif text == "/help":
                self.add("moska", "Commands\n" + "\n".join(
                    f"{command}  {description}" for command, description in COMMANDS.items()
                ) + "\n\nWithout a selected model, the interface demo uses mock workers.\n"
                    "Use /model to select the orchestrator and /agents to enable local Codex workers.\n"
                    "Ctrl+T toggles tool activity. Esc pauses the orchestrator; workers continue.\n"
                    "Send follow-ups to steer. /ps lists agents; /stop stops all background agents.\n"
                    "/connect configures the orchestrator provider; /agents configures local workers.\n"
                    "/connect accounts lists saved accounts; /connect 1 selects one.\n"
                    "https://python-prompt-toolkit.readthedocs.io/")
            elif text.startswith("/"):
                self.add("system", "Unknown command. Type / to see available commands.")
            else:
                self.submit_message(text)

        @keys.add("escape", filter=Condition(lambda: self.picker is not None))
        def dismiss(event):
            if not self.pick_future.done():
                self.pick_future.set_result(None)

        @keys.add("escape", filter=Condition(
            lambda: self.picker is None and self.task is not None and not self.task.done()
        ))
        def interrupt(event):
            self.interrupt_response()

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
            if self.transcript.selected_text:
                self.copy_selection(self.transcript.selected_text)
            elif self.task and not self.task.done():
                self.interrupt_response()
            else:
                self.composer.buffer.reset()

        @keys.add("pageup")
        def scroll_up(event):
            self.scroll_transcript(-10)

        @keys.add("pagedown")
        def scroll_down(event):
            self.scroll_transcript(10)

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
                ("class:muted", "  Enter send · Shift+Enter newline · PgUp/PgDn scroll · Ctrl+T details · Esc pause"),
            ]), height=1),
            VSplit([
                Window(FormattedTextControl(lambda: [("class:accent", "  " + self.activity)])),
                Window(FormattedTextControl(lambda: [("class:accent", self.worker_counter)]), width=28),
            ], height=1),
        ])
        self.root = FloatContainer(body, floats=[])
        app_options.setdefault("refresh_interval", 1)
        self.app = Application(
            layout=Layout(self.root, focused_element=self.composer),
            key_bindings=keys, full_screen=True, mouse_support=True,
            # AIRI palette: printed hex values in AIJ_Demo_color_theme.png.
            style=Style.from_dict({
                "": "bg:#161c27 #ffffff",
                "brand": "#2fbead bold",
                "accent": "#84e0d5",
                "muted": "#909393",
                "tool": "bold",
                "agent": "#2fbead bold",
                "you": "#ffffff bold",
                "moska": "#2fbead bold",
                "system": "#ffffff",
                "link": "#92c7ff underline",
                "markdown-heading": "#2fbead bold",
                "markdown-quote": "#909393 italic",
                "markdown-code": "#ffffff",
                "composer": "bg:#005a60 #ffffff",
                "completion-menu": "bg:#005a60 #ffffff",
                "completion-menu.completion.current": "bg:#066d74 #ffffff noreverse",
                "completion-menu.meta.completion": "bg:#005a60 #ffffff",
                "completion-menu.meta.completion.current": "bg:#066d74 #ffffff",
                "scrollbar.background": "bg:#005a60",
                "scrollbar.button": "bg:#2fbead",
                "scrollbar.arrow": "#ffffff",
                "dialog": "bg:#161c27 #ffffff",
                "dialog.body": "bg:#161c27 #ffffff",
                "dialog frame.border": "#2fbead",
                "dialog frame.label": "bg:#005a60 #3debcb bold",
                "dialog shadow": "bg:#161c27",
                "dialog.body shadow": "bg:#161c27",
                "radio-selected": "bg:#066d74 #ffffff",
                "radio-checked": "#84e0d5",
                "agent-option": "#ffffff",
                "agent-option.focused": "bg:#066d74",
                "agent-option.disabled": "#909393 italic",
                "agent-option.focused agent-option.disabled": "bg:#161c27 underline",
                "selected-text": "bg:#005a60 #ffffff",
            }),
            **app_options,
        )
        if self.session and self.session.transcript():
            self.restore_session()
        else:
            self.add("moska", "What would you like to work on?\n"
                     "Use /model to choose the orchestrator and /agents to configure local workers.")
        try:
            self.account = self.connection.current_account()
            self.activate_model()
            self.activity = self.idle_activity
        except ConnectionError as error:
            self.account = None
            self.activate_model()
            self.add("system", str(error))

    @property
    def idle_activity(self):
        if self.model:
            workers = "local workers" if self.worker_runtime else "mock workers"
            return f"{self.model} · {self.reasoning_effort or 'default'} reasoning · {workers}"
        if self.account and self.account.plan_enabled:
            return f"{self.provider.account_name} connected · interface demo · mock workers"
        return "interface demo · mock workers"

    def copy_selection(self, text):
        async def copy():
            try:
                await copy_text(text)
            except (OSError, RuntimeError, TimeoutError) as error:
                self.add("system", str(error))
        self.app.create_background_task(copy())

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
        if self.session and not self._restoring:
            for progress in self.progress:
                if progress.storage_id is not None and progress.persisted_revision != progress.revision:
                    self.session.update(progress.storage_id, progress.snapshot())
                    progress.persisted_revision = progress.revision
        if self.follow_output:
            self.cursor_line = self.last_line
        self.app.invalidate()

    @property
    def last_line(self):
        return max(0, sum(fragment[1].count("\n") for fragment in self.render_transcript()) - 1)

    def scroll_transcript(self, lines):
        self.cursor_line = max(0, min(self.last_line, self.cursor_line + lines))
        self.follow_output = lines > 0 and self.cursor_line == self.last_line
        self.app.invalidate()

    def add(self, role, text):
        text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
        if self.session and not self._restoring:
            self.session.append('message', {'role': role, 'text': text})
        self.fragments.append((f"class:{role}", role + "\n"))
        self.fragments.extend(markdown_fragments(text) if role == 'moska' else plain_fragments(text))
        self.fragments.append(("", "\n\n"))
        if self.follow_output:
            self.cursor_line = self.last_line
        self.app.invalidate()

    def restore_session(self):
        self._restoring = True
        try:
            for row, kind, payload in self.session.transcript():
                if kind == 'message':
                    self.add(payload['role'], payload['text'])
                    if payload['role'] == 'you':
                        self.composer.buffer.history.append_string(payload['text'])
                elif kind == 'clear':
                    self.fragments.clear()
                    self.progress.clear()
                elif kind == 'progress':
                    progress = ResponseProgress.restore(len(self.fragments), self.refresh_progress, payload)
                    progress.storage_id = row
                    if not progress.finished:
                        progress.finished = True
                        progress.stage = 'Interrupted'
                        progress.events.append({'tool': None, 'text': 'Interrupted by Moska exit', 'details': ''})
                        self.session.update(row, progress.snapshot())
                    self.fragments.append(('', ''))
                    self.progress.append(progress)
        finally:
            self._restoring = False
        self.refresh_progress()

