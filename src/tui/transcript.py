"""Selectable formatted transcript with native clipboard integration."""

import asyncio
import shutil
import sys

from prompt_toolkit.layout import FormattedTextControl
from prompt_toolkit.layout.controls import UIContent
from prompt_toolkit.mouse_events import MouseButton, MouseEventType


async def copy_text(text):
    if sys.platform == 'darwin':
        command = ['pbcopy']
    elif sys.platform == 'win32':
        command = ['clip.exe']
    else:
        command = next((args for args in (
            ['wl-copy'], ['xclip', '-selection', 'clipboard'], ['xsel', '--clipboard', '--input'],
        ) if shutil.which(args[0])), None)
        if command is None:
            raise RuntimeError('No clipboard helper found. Install wl-clipboard, xclip, or xsel.')
    process = await asyncio.create_subprocess_exec(
        *command, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        await asyncio.wait_for(process.communicate(text.encode('utf-8')), 3)
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise
    if process.returncode:
        raise RuntimeError('Could not copy the selection to the clipboard.')


class SelectableTranscript(FormattedTextControl):
    def __init__(self, text, changed, copy, *, scroll=None, **kwargs):
        self.source = text
        self.changed = changed
        self.copy = copy
        self.scroll = scroll
        self.anchor = None
        self.end = None
        self.dragging = False
        self.snapshot = ''
        self._prepared_key = None
        self._prepared_content = None
        super().__init__(self.formatted_text, **kwargs)

    def create_content(self, width, height):
        # FormattedTextControl splits every fragment into lines before consulting
        # its content cache. Layout calls this several times per redraw, and
        # scrolling changes the cursor without changing any of those lines.
        fragments = self._get_formatted_text_cached()
        key = (tuple(fragments), width)
        if key != self._prepared_key:
            self._prepared_content = super().create_content(width, height)
            self._prepared_key = key
            cursor = self._prepared_content.cursor_position
        else:
            cursor = (self.get_cursor_position() if self.get_cursor_position
                      else self._prepared_content.cursor_position)
        self._fragments = fragments
        # Reuse lines, but keep cursor and per-render line-height caches local.
        # Retaining the latter would accumulate entries for every render counter.
        prepared = self._prepared_content
        return UIContent(
            get_line=prepared.get_line, line_count=prepared.line_count,
            cursor_position=cursor, menu_position=prepared.menu_position,
            show_cursor=prepared.show_cursor,
        )

    @staticmethod
    def offset(text, point):
        lines = text.splitlines(keepends=True)
        if point.y >= len(lines):
            return len(text)
        return sum(len(line) for line in lines[:point.y]) + min(point.x, len(lines[point.y].rstrip('\n')))

    @property
    def selected_text(self):
        if self.anchor is None or self.end is None:
            return ''
        start, end = sorted((self.anchor, self.end))
        return self.snapshot[start:end]

    def formatted_text(self):
        fragments = self.source()
        if self.anchor is None or self.end is None or self.anchor == self.end:
            return fragments
        start, end = sorted((self.anchor, self.end))
        result, offset = [], 0
        for fragment in fragments:
            style, text, *handler = fragment
            length = len(text)
            left, right = max(0, start - offset), min(length, end - offset)
            if left < right:
                if left:
                    result.append((style, text[:left], *handler))
                result.append((style + ' class:selected-text', text[left:right], *handler))
                if right < length:
                    result.append((style, text[right:], *handler))
            else:
                result.append(fragment)
            offset += length
        return result

    def mouse_handler(self, event):
        if self.scroll and event.event_type in (MouseEventType.SCROLL_UP, MouseEventType.SCROLL_DOWN):
            self.scroll(-3 if event.event_type == MouseEventType.SCROLL_UP else 3)
            return None
        if event.event_type == MouseEventType.MOUSE_DOWN and event.button == MouseButton.LEFT:
            self.snapshot = ''.join(fragment[1] for fragment in self.source())
            self.anchor = self.end = self.offset(self.snapshot, event.position)
            self.dragging = True
            self.changed()
            return None
        if event.event_type == MouseEventType.MOUSE_MOVE and self.dragging:
            self.end = self.offset(self.snapshot, event.position)
            self.changed()
            return None
        if event.event_type == MouseEventType.MOUSE_UP and self.dragging:
            self.end = self.offset(self.snapshot, event.position)
            self.dragging = False
            self.changed()
            if self.selected_text:
                self.copy(self.selected_text)
                return None
        return super().mouse_handler(event)
