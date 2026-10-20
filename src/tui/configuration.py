"""Provider/model selection and on-device agent configuration dialogs."""

import asyncio
from uuid import uuid4

from prompt_toolkit.layout import Float, HSplit
from prompt_toolkit.widgets import Dialog, Label, RadioList

from adapters.agents import AgentSettingsError, scan_agents
from adapters.providers import MODEL_PROVIDERS, ProviderError as ConnectionError
from orchestrator.demo import DemoCoordinator
from .agents import AgentSelector


class ConfigurationDialogs:
    def activate_model(self):
        self.model = self.connection.selected_model() if self.account and self.account.plan_enabled else None
        self.reasoning_effort = self.connection.selected_reasoning_effort() if self.model else None
        if self.agent_factory:
            demo = not bool(self.model)
            if self.session:
                mode = 'demo' if demo else 'real'
                threads = self.session.meta('thread_ids', {})
                if mode not in threads:
                    threads[mode] = self.config['configurable']['thread_id'] if demo == self.demo_mode else str(uuid4())
                self.config = {'configurable': {'thread_id': threads[mode]}}
                self.session.set_meta('thread_ids', threads)
            elif demo != self.demo_mode:
                # Demo checkpoints contain fictitious workers, not Codex threads.
                self.config = {"configurable": {"thread_id": str(uuid4())}}
            self.demo_mode = demo
            model = self.provider.create_model(self.connection, self.model, self.reasoning_effort) if self.model else DemoCoordinator()
            self.agent = self.agent_factory(model)
        if self.session:
            self.session.set_meta('thread_id', self.config['configurable']['thread_id'])
            self.session.set_meta('demo_mode', self.demo_mode)
            self.session.event('model_selection', {'provider': self.provider.provider_id,
                                                  'model': self.model, 'reasoning_effort': self.reasoning_effort})

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
                     + ("Local workers are configured with /agents." if self.worker_runtime else "Subagents use mock workers."))
        except asyncio.CancelledError:
            raise
        except ConnectionError as error:
            self.add("system", str(error))
        finally:
            self.activity = self.idle_activity
            self.app.invalidate()

    async def configure_agents(self):
        self.activity = "scanning for on-device agents…"
        self.app.invalidate()
        try:
            entries = await asyncio.to_thread(scan_agents)
            enabled = self.agent_settings.enabled()
            self.pick_future = asyncio.get_running_loop().create_future()

            def accept(selected):
                if not self.pick_future.done():
                    self.pick_future.set_result(selected)

            self.picker = AgentSelector(entries, enabled, accept, self.app.invalidate)
            dialog = Float(content=Dialog(title="On-device agents", width=78, body=self.picker))
            self.root.floats.append(dialog)
            self.app.layout.focus(self.picker.control)
            self.activity = "configure on-device agents"
            self.app.invalidate()
            try:
                selected = await self.pick_future
            finally:
                self.root.floats.remove(dialog)
                self.picker = None
                self.pick_future = None
                self.app.layout.focus(self.composer)
            if selected is not None:
                self.agent_settings.save(selected)
                names = [entry.spec.name for entry in entries if entry.spec.agent_id in selected]
                self.add("system", "Local agents: " + (", ".join(names) if names else "none enabled")
                         + ".\nSaved. Workers use Moska's launch directory.")
        except AgentSettingsError as error:
            self.add("system", str(error))
        finally:
            self.activity = self.idle_activity
            self.app.invalidate()

    async def connect(self, option=""):
        self.activity = "connecting model provider · Esc cancel"
        self.app.invalidate()
        try:
            provider = self.provider
            parts = option.split(maxsplit=1)
            if parts and parts[0] in MODEL_PROVIDERS:
                provider = MODEL_PROVIDERS[parts[0]]
                option = parts[1] if len(parts) > 1 else ""
            elif not option and len(MODEL_PROVIDERS) > 1:
                chosen = await self.pick(
                    "Orchestrator provider", [(spec.provider_id, spec.name) for spec in MODEL_PROVIDERS.values()],
                    self.provider.provider_id,
                )
                if chosen is None:
                    return
                provider = MODEL_PROVIDERS[chosen]
            connection = self.connection if provider == self.provider else provider.create_connection()
            accounts = connection.accounts()
            if option == "accounts":
                lines = [
                    f"{index}. {account.email or provider.account_name + ' account'}"
                    + (" · active" if provider == self.provider and self.account and self.account.client_id == account.client_id else "")
                    for index, account in enumerate(accounts, 1)
                ]
                self.add("system", "\n".join(lines) + "\nUse /connect <number> or /connect new."
                         if lines else "No saved accounts. Use /connect to sign in.")
                return
            client_id = None
            if option.isdecimal() and 1 <= int(option) <= len(accounts):
                client_id = accounts[int(option) - 1].client_id
            elif option not in ("", "new"):
                self.add("system", "Use /connect, /connect new, or /connect accounts.")
                return

            def notify(url):
                self.add("system", provider.connect_label + "\n" + provider.connect_instructions
                         + "\nIf the browser doesn't open, click this link:\n" + url)

            previous = self.account
            account = await connection.sign_in(notify, new_account=option == "new", client_id=client_id)
            if provider != self.provider or not previous or previous.client_id != account.client_id:
                if not self.session:
                    self.config = {"configurable": {"thread_id": str(uuid4())}}
            self.provider, self.connection, self.account = provider, connection, account
            self.activate_model()
            if self.account.plan_enabled:
                self.add("system", f"Connected to {provider.account_name} as {self.account.email or 'your account'}.\n"
                         + provider.access_label + " is enabled. Use /model to choose the orchestrator model.")
            else:
                self.add("system", "Signed in, but model access wasn't granted.\nRun /connect again to enable access.")
        except asyncio.CancelledError:
            self.add("system", "Sign-in cancelled. Your previous connection is unchanged.")
            raise
        except ConnectionError as error:
            self.add("system", str(error))
        finally:
            self.activity = self.idle_activity
            self.app.invalidate()

