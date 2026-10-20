import unittest
from unittest.mock import AsyncMock, Mock, patch

from prompt_toolkit.data_structures import Point
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

from tui.transcript import SelectableTranscript, copy_text


def mouse(kind, x, y):
    return MouseEvent(Point(x, y), kind, MouseButton.LEFT, frozenset())


class SelectionTests(unittest.TestCase):
    def test_drag_selects_multiline_text_and_copies_on_release(self):
        copied = []
        control = SelectableTranscript(lambda: [('', 'hello\nworld')], lambda: None, copied.append)
        control.mouse_handler(mouse(MouseEventType.MOUSE_DOWN, 0, 0))
        control.mouse_handler(mouse(MouseEventType.MOUSE_MOVE, 4, 1))
        self.assertEqual(control.selected_text, 'hello\nworl')
        self.assertEqual(copied, [])
        self.assertTrue(any('selected-text' in item[0] for item in control.formatted_text()))
        control.mouse_handler(mouse(MouseEventType.MOUSE_UP, 4, 1))
        self.assertEqual(copied, ['hello\nworl'])

    def test_reverse_drag_copies_plain_text_without_markup(self):
        copied = []
        control = SelectableTranscript(lambda: [('bold', 'hello\n'), ('class:agent', 'world')], lambda: None, copied.append)
        control.mouse_handler(mouse(MouseEventType.MOUSE_DOWN, 5, 1))
        control.mouse_handler(mouse(MouseEventType.MOUSE_UP, 2, 0))
        self.assertEqual(copied, ['llo\nworld'])

    def test_click_without_drag_preserves_links(self):
        clicked, copied = [], []
        control = SelectableTranscript(lambda: [('', 'link', clicked.append)], lambda: None, copied.append)
        control.create_content(80, 20)
        control.mouse_handler(mouse(MouseEventType.MOUSE_DOWN, 1, 0))
        control.mouse_handler(mouse(MouseEventType.MOUSE_UP, 1, 0))
        self.assertEqual(copied, [])
        self.assertEqual(len(clicked), 1)


class ClipboardTests(unittest.IsolatedAsyncioTestCase):
    async def test_mac_clipboard_uses_pbcopy_without_shell(self):
        process = Mock(returncode=0, communicate=AsyncMock(return_value=(None, None)))
        with patch('tui.transcript.sys.platform', 'darwin'), patch('tui.transcript.asyncio.create_subprocess_exec', new_callable=AsyncMock, return_value=process) as create:
            await copy_text('selected text')
        self.assertEqual(create.call_args.args, ('pbcopy',))
        process.communicate.assert_awaited_once_with(b'selected text')
