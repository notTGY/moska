"""Adapt local Codex threads/turns to the decomposer's existing worker protocol.

HTTP-shaped requests stay in process; no socket or worker server is exposed.
The external decomposer and its evolved tools/prompts remain untouched.
"""

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import httpx

from .codex import CodexError, CodexServer
from .registry import AgentSettingsError, require_codex


class CodexWorkers:
    agent_type = {
        'agent_type_id': 'codex', 'assistant_id': 'codex',
        'description': 'Local Codex coding agent. Works in the same workspace as Moska.',
        'url': 'http://codex.local',
    }

    def __init__(self, workspace: Path, settings, *, on_event=None):
        self.workspace = workspace.resolve()
        self.settings = settings
        self.on_event = on_event or (lambda thread, item: None)
        self.server = None
        self.start_lock = asyncio.Lock()
        self.threads = {}
        self.runs = {}
        self.active = {}
        self.start_tasks = set()
        self.closing = False

    async def _server(self):
        if 'codex' not in self.settings.enabled():
            raise CodexError('Codex is disabled. Enable it with /agents first.')
        async with self.start_lock:
            if self.closing:
                raise CodexError('Codex workers are stopping.')
            if self.server is None:
                executable = await asyncio.to_thread(require_codex)
                self.server = CodexServer(executable, self.workspace,
                                          on_event=self._event, on_disconnect=self._disconnected)
            await self.server.start()
            return self.server

    def _thread_options(self):
        return {'cwd': str(self.workspace), 'approvalPolicy': 'never', 'sandbox': 'workspace-write'}

    async def handle(self, request):
        try:
            return await self._handle(request)
        except (CodexError, AgentSettingsError, ValueError, KeyError) as error:
            return httpx.Response(400, json={'detail': str(error)})

    async def _handle(self, request):
        parts = request.url.path.strip('/').split('/')
        body = json.loads(request.content) if request.content else {}
        if parts == ['threads'] and request.method == 'POST':
            server = await self._server()
            result = await server.request('thread/start', self._thread_options())
            thread_id = result['thread']['id']
            self.threads[thread_id] = {'messages': [], 'error': None}
            return httpx.Response(200, json={'thread_id': thread_id})

        thread_id = parts[1]
        thread = self.threads[thread_id]
        if parts[2:] == ['state']:
            checkpoint = {'checkpoint_id': thread_id} if thread['messages'] else None
            return httpx.Response(200, json={'checkpoint': checkpoint})
        if len(parts) == 2 and request.method == 'GET':
            return httpx.Response(200, json={'thread_id': thread_id, 'error': thread['error']})
        if parts[2:] == ['copy']:
            if thread_id in self.active:
                raise CodexError('Wait for the active Codex turn before forking.')
            server = await self._server()
            result = await server.request('thread/fork', {'threadId': thread_id, **self._thread_options()})
            fork_id = result['thread']['id']
            self.threads[fork_id] = deepcopy(thread)
            return httpx.Response(200, json={'thread_id': fork_id})
        if parts[2:] == ['runs'] and request.method == 'POST':
            if thread_id in self.active:
                raise CodexError('This Codex agent already has an active turn.')
            server = await self._server()
            run_id = str(uuid4())
            before = thread['messages'] + body['input']['messages']
            run = {'run_id': run_id, 'thread_id': thread_id, 'status': 'pending',
                   'turn_id': None, 'before': before, 'messages': [], 'items': {}, 'error': None}
            self.runs[run_id] = run
            self.active[thread_id] = run
            thread['error'] = None
            prompt = body['input']['messages'][-1]['content']
            task = asyncio.create_task(self._start_turn(server, run, prompt))
            self.start_tasks.add(task)
            task.add_done_callback(self.start_tasks.discard)
            await asyncio.shield(task)
            # Always acknowledge a start as pending, including very fast completions.
            return httpx.Response(200, json={'run_id': run_id, 'status': 'pending'})
        if parts[2:3] == ['runs'] and request.method == 'GET':
            run = self.runs[parts[3]]
            if run['thread_id'] != thread_id:
                raise CodexError('Run does not belong to this Codex agent.')
            return httpx.Response(200, json={key: run[key] for key in ('run_id', 'thread_id', 'status')})
        if parts[2:] == ['history']:
            run = self.runs[body['metadata']['run_id']]
            if run['thread_id'] != thread_id:
                raise CodexError('Run does not belong to this Codex agent.')
            return httpx.Response(200, json=[
                {'values': {'messages': run['before'] + run['messages']}, 'metadata': {'source': 'loop'}},
                {'values': {'messages': run['before']}, 'metadata': {'source': 'input'}},
            ])
        return httpx.Response(404, json={'detail': 'Unsupported local worker operation.'})

    async def _start_turn(self, server, run, prompt):
        try:
            result = await server.request('turn/start', {
                'threadId': run['thread_id'], 'input': [{'type': 'text', 'text': prompt}],
                'cwd': str(self.workspace), 'approvalPolicy': 'never',
                'sandboxPolicy': {'type': 'workspaceWrite', 'writableRoots': [str(self.workspace)],
                                  'networkAccess': False, 'excludeTmpdirEnvVar': True, 'excludeSlashTmp': True},
            })
            run['turn_id'] = result['turn']['id']
            if run['status'] == 'pending':
                run['status'] = 'running'
            if result['turn'].get('status') in ('completed', 'failed', 'interrupted'):
                self._complete(run, result['turn'])
        except Exception as error:
            self._complete(run, {'status': 'failed', 'error': {'message': str(error)}})

    def _event(self, method, params):
        run = self.active.get(params.get('threadId'))
        if not run:
            return
        turn_id = params.get('turnId') or params.get('turn', {}).get('id')
        if run['turn_id'] is not None and turn_id is not None and turn_id != run['turn_id']:
            return
        if method == 'turn/started':
            run['turn_id'] = params['turn']['id']
            run['status'] = 'running'
        elif method == 'item/completed':
            item = params['item']
            run['items'][item['id']] = item
            self.on_event(run['thread_id'], item)
        elif method == 'item/started':
            self.on_event(run['thread_id'], params['item'])
        elif method == 'turn/completed':
            self._complete(run, params['turn'])

    def _complete(self, run, turn):
        if run['status'] in ('success', 'error', 'interrupted'):
            return
        for item in turn.get('items', []):
            run['items'][item['id']] = item
        messages = []
        answers = []
        for item in run['items'].values():
            kind = item['type']
            if kind == 'agentMessage':
                answers.append(item)
                messages.append({'type': 'ai', 'content': item['text'], 'tool_calls': []})
            elif kind in ('commandExecution', 'fileChange', 'mcpToolCall', 'dynamicToolCall', 'collabToolCall', 'webSearch', 'imageView'):
                messages.append({'type': 'ai', 'content': '', 'tool_calls': [{
                    'id': item['id'], 'name': kind,
                    'args': {k: v for k, v in item.items() if k not in ('aggregatedOutput', 'result', 'id', 'type')},
                }]})
                messages.append({'type': 'tool', 'tool_call_id': item['id'],
                                 'content': json.dumps(item, ensure_ascii=False), 'tool_calls': []})
        finals = [answer for answer in answers if answer.get('phase') == 'final_answer']
        candidates = finals or [answer for answer in answers if answer.get('phase') != 'commentary']
        final = candidates[-1]['text'] if candidates else ''
        if final and (not messages or messages[-1].get('content') != final or messages[-1].get('tool_calls')):
            messages.append({'type': 'ai', 'content': final, 'tool_calls': []})
        status = {'completed': 'success', 'interrupted': 'interrupted', 'failed': 'error'}.get(turn['status'], 'error')
        error = turn.get('error')
        if status == 'success' and not final.strip():
            status, error = 'error', {'message': 'Codex completed without a final response.'}
        run.update(status=status, messages=messages, error=error)
        thread = self.threads[run['thread_id']]
        thread['messages'] = run['before'] + messages
        thread['error'] = error
        self.active.pop(run['thread_id'], None)

    def _disconnected(self, error):
        for run in list(self.active.values()):
            self._complete(run, {'status': 'failed', 'error': {'message': str(error)}})

    async def interrupt_all(self):
        async def interrupt_run(run):
            if not run['turn_id'] or run.get('interrupt_requested'):
                return
            try:
                await self.server.request('turn/interrupt', {
                    'threadId': run['thread_id'], 'turnId': run['turn_id'],
                }, timeout=5)
                run['interrupt_requested'] = True
            except CodexError as error:
                # Native turn/start can acknowledge a queued turn before it is
                # interruptible. Retry until it starts or a terminal event arrives.
                if 'no active turn' not in str(error).lower():
                    raise

        async def interrupt():
            if self.start_tasks:
                await asyncio.gather(*list(self.start_tasks), return_exceptions=True)
            if self.server and self.server.ready:
                # An interrupt ack is not completion: await terminal notifications.
                while self.active:
                    await asyncio.gather(*(interrupt_run(run) for run in list(self.active.values())))
                    await asyncio.sleep(.1)
        try:
            await asyncio.wait_for(interrupt(), 7)
        except (TimeoutError, CodexError, OSError):
            for run in list(self.active.values()):
                self._complete(run, {'status': 'interrupted'})
            if self.server:
                await self.server.close()
        finally:
            for run in list(self.active.values()):
                self._complete(run, {'status': 'interrupted'})

    async def close(self):
        self.closing = True
        await self.interrupt_all()
        if self.server:
            await self.server.close()
        tasks = list(self.start_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
