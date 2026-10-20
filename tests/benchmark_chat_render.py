"""Run with .venv/bin/python tests/benchmark_chat_render.py [--profile].

Compare the stock control's preparation path with Moska's cache in the same
100-message chat. DummyOutput measures layout/render CPU, not terminal I/O.
"""

import asyncio
from contextlib import nullcontext
import cProfile
import pstats
import statistics
import sys
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.layout import FormattedTextControl
from prompt_toolkit.output import DummyOutput

from tui.app import ChatUI
from tui.transcript import SelectableTranscript


async def measure(cached, profile=False):
    connection = SimpleNamespace(current_account=lambda: None)
    control = (nullcontext() if cached else patch.object(
        SelectableTranscript, 'create_content', FormattedTextControl.create_content))
    with control, patch('adapters.providers.registry.OpenAIConnection', return_value=connection), create_pipe_input() as pipe:
        ui = ChatUI(None, input=pipe, output=DummyOutput())
        # Replace the greeting with exactly 100 messages.
        ui.fragments.clear()
        for i in range(100):
            ui.add('moska', f'## Message {i}\n' + (
                'Some **formatted** text with `code` and a [link](https://example.com).\n' * 12))
        task = asyncio.create_task(ui.app.run_async())
        try:
            await asyncio.sleep(.05)
            profiler = cProfile.Profile() if profile else None
            if profiler:
                profiler.enable()
            start = perf_counter()
            for i in range(100):
                ui.scroll_transcript(-1 if i < 50 else 1)
                ui.app._redraw()
            elapsed = perf_counter() - start
            if profiler:
                profiler.disable()
                pstats.Stats(profiler).strip_dirs().sort_stats('cumtime').print_stats(15)
            return elapsed
        finally:
            ui.app.exit()
            await task


async def main():
    profile = '--profile' in sys.argv
    results = {False: [], True: []}
    # Alternate order to reduce warm-up/order bias.
    for trial in range(5):
        for cached in ((False, True) if trial % 2 == 0 else (True, False)):
            results[cached].append(await measure(cached, profile and trial == 0))
    for cached, samples in results.items():
        print(f'{"cached" if cached else "stock"}: median {statistics.median(samples) * 1000:.2f} ms / '
              f'100 scroll+redraw operations; samples {[round(s * 1000, 2) for s in samples]}')
    print(f'Speedup: {statistics.median(results[False]) / statistics.median(results[True]):.2f}x')


if __name__ == '__main__':
    asyncio.run(main())
