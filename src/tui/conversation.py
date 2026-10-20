"""Orchestrator execution and immediate follow-up steering."""

import asyncio
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage

from .progress import ProgressCallbacks, ResponseProgress


class Conversation:
    def submit_message(self, text):
        self.follow_output = True
        self.add('you', text)
        message = {'id': 'user-' + str(uuid4()), 'content': text}
        self.pending_messages.append(message)
        if self.session:
            queued = self.session.meta('pending_messages', [])
            self.session.set_meta('pending_messages', [*queued, message])
        if self.responding and self.task and not self.task.done():
            self.restart_response = True
            self.activity = 'steering orchestrator…'
            if not self.preparing_turn and not self.task.cancelling():
                self.task.cancel()
        else:
            self._launch_response()

    def _launch_response(self):
        self.responding = True
        self.restart_response = False
        self.task = self.app.create_background_task(self.respond())

        def finished(task):
            if self.task is not task:
                return
            self.responding = False
            if (self.pending_messages and self.restart_response and self.app.is_running
                    and not self.app.is_done and not (self.stop_task and not self.stop_task.done())):
                self._launch_response()

        self.task.add_done_callback(finished)

    def interrupt_response(self):
        self.restart_response = False
        if self.task and not self.task.done() and not self.task.cancelling():
            self.task.cancel()

    async def respond(self, prompt=None):
        progress = ResponseProgress(len(self.fragments), self.refresh_progress)
        self.fragments.append(('', ''))
        self.progress.append(progress)
        if self.session:
            progress.storage_id = self.session.append('progress', progress.snapshot())
        progress.record('Message received', 'Preparing request')
        self.activity = 'working… · Esc pause orchestrator · /stop agents'
        callbacks = ProgressCallbacks(progress, self.model or 'scripted demo')
        try:
            self.preparing_turn = True
            if self.prepare_turn:
                await self.prepare_turn(self.agent, self.config)
            if self.session and prompt is None:
                queued = self.session.meta('pending_messages', [])
                state = await self.agent.aget_state(self.config)
                existing = {message.id for message in state.values.get('messages', [])}
                self.pending_messages = [message for message in queued if message['id'] not in existing]
                self.session.set_meta('pending_messages', self.pending_messages)
            messages = [HumanMessage(content=prompt)] if prompt is not None else [
                HumanMessage(**message) for message in self.pending_messages]
            if prompt is None:
                self.pending_messages.clear()
            self.preparing_turn = False
            self.restart_response = False
            async for update in self.agent.astream(
                {'messages': messages, 'agents': {}, 'agent_runs': {}},
                config={**self.config, 'callbacks': [callbacks]}, stream_mode='updates',
            ):
                if self.session and prompt is None:
                    sent = {message.id for message in messages}
                    queued = self.session.meta('pending_messages', [])
                    self.session.set_meta('pending_messages', [message for message in queued if message['id'] not in sent])
                for value in update.values():
                    if not isinstance(value, dict):
                        continue
                    if self.session:
                        self.session.event('graph_update', value)
                    progress.state_update(value)
                    for message in value.get('messages', []):
                        if isinstance(message, AIMessage) and message.content and not message.tool_calls:
                            self.add('moska', message.text)
                self.app.invalidate()
            progress.finish('Responded')
        except asyncio.CancelledError:
            if self.restart_response and self.pending_messages:
                progress.finish('Steered')
            else:
                progress.finish('Interrupted')
                self.add('system', 'Orchestrator paused. Background agents continue.' if self.worker_runtime
                         and self.worker_runtime.running_count else 'Orchestrator paused.')
            raise
        except Exception as error:
            progress.finish('Failed')
            self.add('system', f'Error: {error}')
        finally:
            self.preparing_turn = False
            self.activity = self.idle_activity
            self.app.invalidate()
