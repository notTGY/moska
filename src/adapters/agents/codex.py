"""Small async client for the installed Codex App Server's stdio protocol."""

import asyncio
import json
import os
import signal
from pathlib import Path


class CodexError(RuntimeError):
    pass


class CodexServer:
    def __init__(self, executable, workspace: Path, *, on_event=None, on_disconnect=None):
        self.executable = executable
        self.workspace = workspace.resolve()
        self.on_event = on_event or (lambda method, params: None)
        self.on_disconnect = on_disconnect or (lambda error: None)
        self.process = None
        self.reader = None
        self.stderr_reader = None
        self.pending = {}
        self.next_id = 0
        self.start_lock = asyncio.Lock()
        self.write_lock = asyncio.Lock()
        self.ready = False

    async def start(self):
        async with self.start_lock:
            if self.ready:
                return
            if self.process is not None:
                raise CodexError('Codex disconnected. Restart Moska to reconnect.')
            try:
                self.process = await asyncio.create_subprocess_exec(
                    self.executable, 'app-server', '--listen', 'stdio://',
                    '-c', 'sandbox_mode="workspace-write"',
                    '-c', 'approval_policy="never"',
                    '-c', 'sandbox_workspace_write.network_access=false',
                    '-c', 'sandbox_workspace_write.writable_roots=[]',
                    cwd=self.workspace, stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    limit=16 * 1024 * 1024, start_new_session=os.name == 'posix',
                )
                self.reader = asyncio.create_task(self._read())
                self.stderr_reader = asyncio.create_task(self._drain_stderr())
                await self.request('initialize', {
                    'clientInfo': {'name': 'moska', 'title': 'Moska', 'version': '0.1.0'},
                    'capabilities': {'experimentalApi': False},
                })
                await self._write({'method': 'initialized', 'params': {}})
                self.ready = True
            except BaseException:
                await self.close()
                raise

    async def request(self, method, params, *, timeout=30):
        if self.process is None or self.process.returncode is not None:
            raise CodexError('Codex is not running. Install Codex and run codex login first.')
        self.next_id += 1
        request_id = self.next_id
        future = asyncio.get_running_loop().create_future()
        self.pending[request_id] = future
        try:
            await self._write({'id': request_id, 'method': method, 'params': params})
            return await asyncio.wait_for(future, timeout)
        except TimeoutError:
            # A timed-out mutating request may still execute. Closing the private
            # server prevents an untracked turn from continuing in the background.
            await self.close()
            raise CodexError(f'Codex timed out during {method}.') from None
        finally:
            self.pending.pop(request_id, None)

    async def _write(self, message):
        async with self.write_lock:
            self.process.stdin.write((json.dumps(message) + '\n').encode())
            await self.process.stdin.drain()

    async def _read(self):
        error = CodexError('Codex App Server disconnected. Check codex login and restart Moska.')
        try:
            while line := await self.process.stdout.readline():
                message = json.loads(line)
                if 'method' in message:
                    if 'id' in message:
                        await self._deny_request(message)
                    else:
                        self.on_event(message['method'], message.get('params') or {})
                elif future := self.pending.get(message.get('id')):
                    if not future.done():
                        if 'error' in message:
                            future.set_exception(CodexError(message['error'].get('message', 'Codex request failed.')))
                        else:
                            future.set_result(message.get('result', {}))
        except asyncio.CancelledError:
            raise
        except (ValueError, KeyError, TypeError, OSError) as cause:
            error = CodexError(f'Could not read the Codex protocol ({type(cause).__name__}).')
        finally:
            self.ready = False
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(error)
            self.on_disconnect(error)
            if self.process.returncode is None:
                self._signal(signal.SIGTERM)

    async def _deny_request(self, message):
        """Never silently grant approval, extra permissions, credentials or tool execution."""
        method = message['method']
        if method in ('item/commandExecution/requestApproval', 'item/fileChange/requestApproval'):
            result = {'decision': 'decline'}
        elif method == 'item/permissions/requestApproval':
            result = {'permissions': {}, 'scope': 'turn'}
        elif method == 'item/tool/requestUserInput':
            result = {'answers': {}}
        elif method == 'mcpServer/elicitation/request':
            result = {'action': 'decline', 'content': None}
        elif method == 'item/tool/call':
            result = {'success': False, 'contentItems': [{'type': 'inputText', 'text': 'Host tool execution is unavailable in Moska.'}]}
        else:
            await self._write({'id': message['id'], 'error': {'code': -32601, 'message': 'Unsupported host request.'}})
            return
        await self._write({'id': message['id'], 'result': result})

    async def _drain_stderr(self):
        # Drain without mixing runtime diagnostics (or credentials) into the TUI.
        while await self.process.stderr.read(65536):
            pass

    async def close(self):
        self.ready = False
        process = self.process
        if process is None:
            return
        if process.stdin:
            process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), 2)
        except TimeoutError:
            self._signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), 2)
            except TimeoutError:
                self._signal(signal.SIGKILL)
                await process.wait()
        tasks = [task for task in (self.reader, self.stderr_reader) if task]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _signal(self, sig):
        try:
            if os.name == 'posix':
                os.killpg(self.process.pid, sig)
            elif sig == signal.SIGTERM:
                self.process.terminate()
            else:
                self.process.kill()
        except ProcessLookupError:
            pass
