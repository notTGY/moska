"""Moska command-line entry point."""

import asyncio
import argparse
import json
import os
from pathlib import Path

from orchestrator.runtime import chat
from orchestrator.storage import SessionBusyError, SessionStore


def main():
    parser = argparse.ArgumentParser(description='Moska local agent workspace')
    parser.add_argument('--export-session', type=Path, metavar='FILE', help='Export this workspace session as JSON')
    args = parser.parse_args()
    try:
        if args.export_session:
            session = SessionStore(Path.cwd())
            if not session.path.exists():
                parser.error('No saved session exists for this workspace.')
            session.open()
            try:
                fd = os.open(args.export_session, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
                with os.fdopen(fd, 'w') as output:
                    json.dump(session.export(), output, ensure_ascii=False, indent=2)
            finally:
                session.close()
        else:
            asyncio.run(chat())
    except SessionBusyError as error:
        parser.exit(1, str(error) + '\n')
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
