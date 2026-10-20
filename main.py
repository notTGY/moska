"""Moska command-line entry point."""

import asyncio

from orchestrator.runtime import chat


def main():
    try:
        asyncio.run(chat())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
