"""Preserve worker tool acknowledgements when an orchestrator invocation stops."""

import asyncio
import json
import time

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage, messages_from_dict
from langgraph.types import Command

from adapters.agents.context import worker_operation


def decode(value):
    if isinstance(value, dict):
        if '__message__' in value:
            return messages_from_dict([value['__message__']])[0]
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value


class ToolContinuity(AgentMiddleware):
    def __init__(self, storage=None, workers=None):
        self.storage = storage
        self.workers = workers
        self.tasks = {}
        self.results = {}

    async def awrap_tool_call(self, request, handler):
        if request.tool_call['name'] == 'wait':
            return await handler(request)
        thread = request.runtime.config['configurable']['thread_id']
        call = request.tool_call['id']
        key = (thread, call)

        async def execute():
            token = worker_operation.set({'thread_id': thread, 'call_id': call})
            record = {'name': request.tool_call['name'], 'args': request.tool_call['args'], 'status': 'started'}
            self._save(thread, call, record)
            try:
                result = await handler(request)
                record = {**record, 'status': 'done', 'update': result.update if isinstance(result, Command) else {'messages': [result]}}
                self._save(thread, call, record)
                return result
            except Exception as error:
                self._save(thread, call, {**record, 'status': 'error', 'error': str(error)})
                raise
            finally:
                worker_operation.reset(token)

        task = asyncio.create_task(execute())
        self.tasks[key] = task
        task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        # Cancel the orchestrator, not a worker operation whose acknowledgement
        # must still be recorded. The next invocation reconciles its checkpoint.
        return await asyncio.shield(task)

    def _save(self, thread, call, record):
        self.results[(thread, call)] = record
        if self.storage:
            self.storage.save_tool(thread, call, record)
            self.storage.event('orchestrator_tool', {'thread_id': thread, 'call_id': call, **record})

    async def settle(self, thread=None):
        tasks = [task for (owner, _), task in self.tasks.items() if thread is None or owner == thread]
        if tasks:
            await asyncio.shield(asyncio.gather(*tasks, return_exceptions=True))

    async def recover(self, agent, config):
        if not hasattr(agent, 'aget_state'):
            return
        thread = config['configurable']['thread_id']
        await self.settle(thread)
        try:
            snapshot = await agent.aget_state(config)
        except ValueError:  # Lightweight UI demos can omit a checkpointer.
            return
        state = snapshot.values
        if not state:
            return
        records = self.storage.tools(thread) if self.storage else {
            call: record for (owner, call), record in self.results.items() if owner == thread}
        messages = state.get('messages', [])
        resolved = {message.tool_call_id for message in messages if isinstance(message, ToolMessage)}
        update = {'messages': [], 'agents': {}, 'agent_runs': {}}
        for message in messages:
            if not isinstance(message, AIMessage):
                continue
            for call in message.tool_calls:
                if call['id'] in resolved:
                    continue
                record = decode(records.get(call['id'], {}))
                if record.get('status') == 'started' and self.workers:
                    recovered = self.workers.recover_operation({'thread_id': thread, 'call_id': call['id']}, record)
                    if recovered:
                        delta, output = recovered
                        delta['messages'] = [ToolMessage(content=json.dumps(output), tool_call_id=call['id'], name=call['name'])]
                        record = {**record, 'status': 'done', 'update': delta}
                        self._save(thread, call['id'], record)
                if record.get('status') == 'done':
                    delta = record['update']
                    update['messages'].extend(delta.get('messages', []))
                    for field in ('agents', 'agent_runs'):
                        update[field].update(delta.get(field, {}))
                else:
                    if call['name'] == 'wait':
                        text = 'Orchestrator stopped waiting. Background agents may still be running; call wait to collect their results.'
                    elif record.get('status') == 'started':
                        text = 'Moska restarted before recording this operation result. Its outcome is unknown; inspect existing agents before retrying.'
                    else:
                        text = record.get('error') or 'Orchestrator interrupted before this tool completed.'
                    update['messages'].append(ToolMessage(content=text, tool_call_id=call['id'], name=call['name'], status='error'))
        runs = state.get('decomposer_agent_runs', [])
        if runs and runs[-1]['status'] == 'running':
            update['decomposer_agent_runs'] = [*runs[:-1], {**runs[-1], 'status': 'interrupted', 'collected_at': time.time()}]
        if update['messages'] or 'decomposer_agent_runs' in update:
            await agent.aupdate_state(config, update, as_node='tools')
            if self.storage:
                self.storage.event('checkpoint_recovery', update)
        for key in list(self.tasks):
            if key[0] == thread and self.tasks[key].done():
                del self.tasks[key]
                self.results.pop(key, None)
