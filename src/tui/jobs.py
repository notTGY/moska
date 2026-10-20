"""Background worker inspection and explicit shutdown commands."""

import asyncio
import time


def duration(seconds):
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours}h {minutes}m' if hours else (f'{minutes}m {seconds}s' if minutes else f'{seconds}s')


class WorkerCommands:
    @property
    def worker_counter(self):
        count = self.worker_runtime.running_count if self.worker_runtime else 0
        return f'{count} agent' + ('s' if count != 1 else '') + ' running · /ps'

    def show_processes(self):
        rows = self.worker_runtime.processes() if self.worker_runtime else []
        if not rows:
            self.add('system', 'No local agents have been created in this workspace.')
            return
        # Snapshot, so users can select/copy a stable full prompt.
        lines = [self.worker_counter]
        for row in rows:
            status = {'success': 'completed', 'pending': 'starting', 'interrupted': 'stopped'}.get(row['status'], row['status'])
            end = row['finished_at'] if row['finished_at'] is not None else time.time()
            elapsed = duration(end - row['started_at']) if row['started_at'] is not None else '—'
            lines.extend(['', f"{row['agent_type']} {row['agent_id']} · {status} · {elapsed}",
                          'Workspace: ' + row['workspace'], 'Prompt: ' + (row['prompt'] or '—')])
        self.add('system', '\n'.join(lines))

    async def stop_workers(self):
        if not self.worker_runtime:
            self.add('system', 'No local workers are running.')
            return
        # Pause orchestration too, so it cannot immediately replace stopped runs.
        self.restart_response = False
        self.worker_runtime.suspend_starts(True)
        self.activity = 'stopping background agents…'
        self.app.invalidate()
        try:
            running = self.task
            if running and not running.done():
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)
            if self.prepare_turn:
                await self.prepare_turn(self.agent, self.config)
            count = self.worker_runtime.running_count
            await self.worker_runtime.interrupt_all()
            self.add('system', f'Stopped {count} background agent' + ('s.' if count != 1 else '.'))
            if self.session:
                self.session.event('stop_workers', {'count': count})
        finally:
            self.worker_runtime.suspend_starts(False)
            self.activity = self.idle_activity
            self.app.invalidate()
