"""Private, workspace-scoped session storage and inspectable event history."""

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

from langchain_core.messages import BaseMessage, message_to_dict


def encode(value):
    if isinstance(value, BaseMessage):
        return {'__message__': message_to_dict(value)}
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    return value


class SessionBusyError(RuntimeError):
    pass


class SessionStore:
    def __init__(self, workspace, *, data_dir=None):
        self.workspace = Path(workspace).resolve()
        self.session_id = hashlib.sha256(os.fsencode(str(self.workspace))).hexdigest()
        base = Path(data_dir) if data_dir else Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local/share')) / 'moska'
        self.directory = base / 'sessions' / self.session_id
        self.path = self.directory / 'session.sqlite3'
        self.checkpoint_path = self.directory / 'checkpoints.sqlite3'
        self.connection = None
        self.lock = None

    def open(self):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = os.fdopen(os.open(self.directory / 'session.lock', os.O_CREAT | os.O_RDWR, 0o600), 'a+b')
        try:
            if os.name == 'posix':
                import fcntl
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                import msvcrt
                self.lock.write(b'0')
                self.lock.flush()
                self.lock.seek(0)
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            self.lock.close()
            self.lock = None
            raise SessionBusyError('This workspace is already open in another Moska process.') from None
        try:
            for path in (self.path, self.checkpoint_path):
                fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
                os.close(fd)
            self.connection = sqlite3.connect(self.path, timeout=30)
            self.connection.execute('PRAGMA journal_mode=WAL')
            self.connection.executescript('''
                CREATE TABLE IF NOT EXISTS session_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS session_events (
                    id INTEGER PRIMARY KEY, created_at REAL NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS session_transcript (
                    id INTEGER PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS session_workers (
                    runtime TEXT NOT NULL, kind TEXT NOT NULL, id TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(runtime, kind, id));
                CREATE TABLE IF NOT EXISTS session_tools (
                    thread_id TEXT NOT NULL, call_id TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(thread_id, call_id));
            ''')
            self.set_meta('workspace', str(self.workspace))
            self.set_meta('schema_version', 1)
            return self
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _json(payload):
        return json.dumps(encode(payload), ensure_ascii=False)

    def meta(self, key, default=None):
        row = self.connection.execute('SELECT value FROM session_meta WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_meta(self, key, value):
        with self.connection:
            self.connection.execute('INSERT OR REPLACE INTO session_meta VALUES (?,?)', (key, self._json(value)))

    def event(self, kind, payload):
        with self.connection:
            self.connection.execute('INSERT INTO session_events(created_at,kind,payload) VALUES (?,?,?)',
                                    (time.time(), kind, self._json(payload)))

    def append(self, kind, payload):
        with self.connection:
            cursor = self.connection.execute('INSERT INTO session_transcript(kind,payload) VALUES (?,?)',
                                             (kind, self._json(payload)))
            return cursor.lastrowid

    def update(self, row_id, payload):
        with self.connection:
            self.connection.execute('UPDATE session_transcript SET payload=? WHERE id=?', (self._json(payload), row_id))

    def transcript(self):
        return [(row_id, kind, json.loads(payload)) for row_id, kind, payload in
                self.connection.execute('SELECT id,kind,payload FROM session_transcript ORDER BY id')]

    def save_worker(self, runtime, kind, identifier, payload):
        with self.connection:
            self.connection.execute('INSERT OR REPLACE INTO session_workers VALUES (?,?,?,?)',
                                    (runtime, kind, identifier, self._json(payload)))

    def workers(self, runtime, kind):
        return {identifier: json.loads(payload) for identifier, payload in self.connection.execute(
            'SELECT id,payload FROM session_workers WHERE runtime=? AND kind=?', (runtime, kind))}

    def save_tool(self, thread, call, payload):
        with self.connection:
            self.connection.execute('INSERT OR REPLACE INTO session_tools VALUES (?,?,?)',
                                    (thread, call, self._json(payload)))

    def tools(self, thread):
        return {call: json.loads(payload) for call, payload in self.connection.execute(
            'SELECT call_id,payload FROM session_tools WHERE thread_id=?', (thread,))}

    def export(self):
        """Plain JSON view; checkpoint blobs remain in SQLite for exact graph recovery."""
        return {'workspace': str(self.workspace), 'session_id': self.session_id,
                'transcript': [{'id': row, 'kind': kind, **payload} for row, kind, payload in self.transcript()],
                'events': [{'id': row, 'created_at': created, 'kind': kind, 'payload': json.loads(payload)}
                           for row, created, kind, payload in self.connection.execute(
                               'SELECT id,created_at,kind,payload FROM session_events ORDER BY id')],
                'workers': [{'runtime': runtime, 'kind': kind, 'id': identifier, 'payload': json.loads(payload)}
                            for runtime, kind, identifier, payload in self.connection.execute(
                                'SELECT runtime,kind,id,payload FROM session_workers')]}

    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None
        if self.lock:
            if os.name != 'posix':
                import msvcrt
                self.lock.seek(0)
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_UNLCK, 1)
            self.lock.close()
            self.lock = None
