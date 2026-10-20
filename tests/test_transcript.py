import unittest
from unittest.mock import AsyncMock, Mock, patch

from prompt_toolkit import Application
from prompt_toolkit.application.current import set_app
from prompt_toolkit.data_structures import Point
from prompt_toolkit.input import DummyInput
from prompt_toolkit.layout import FormattedTextControl
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.output import DummyOutput

from tui.transcript import SelectableTranscript, copy_text


def mouse(kind, x, y):
    return MouseEvent(Point(x, y), kind, MouseButton.LEFT, frozenset())


class ContentCacheTests(unittest.TestCase):
    def setUp(self):
        self.app = Application(input=DummyInput(), output=DummyOutput())
        self.context = set_app(self.app)
        self.context.__enter__()
        self.addCleanup(self.context.__exit__, None, None, None)

    def render(self, control, width=80):
        self.app.render_counter += 1
        return control.create_content(width, 20)

    def test_100_message_scroll_reuses_lines_and_keeps_cursor_current(self):
        fragments = [('class:moska', f'message {i}\n' + 'text\n' * 12)
                     for i in range(100)]
        position = Point(0, 0)
        control = SelectableTranscript(
            lambda: list(fragments), lambda: None, lambda text: None,
            get_cursor_position=lambda: position, show_cursor=False,
        )
        original = FormattedTextControl.create_content
        with patch.object(FormattedTextControl, 'create_content', autospec=True,
                          side_effect=original) as prepare:
            first = self.render(control)
            for y in range(1, 101):
                position = Point(0, y)
                content = self.render(control)
                # Height changes do not require splitting the transcript either.
                self.assertEqual(control.create_content(80, 10).cursor_position, position)
                self.assertEqual(content.cursor_position, position)
                self.assertIs(content.get_line, first.get_line)
                self.assertEqual(content.line_count, 1301)
                self.assertFalse(content.show_cursor)
            self.assertEqual(prepare.call_count, 1)
        self.assertEqual(first.cursor_position, Point(0, 0))
        first.get_height_for_line(1, 80, None)
        self.assertGreater(content.get_height_for_line(1, 80, lambda *_: ' ' * 79), 1)

    def test_content_width_selection_and_handler_changes_match_base_control(self):
        clicked = []
        fragments = [('class:link', 'hello', clicked.append), ('', '\nworld\n')]
        control = SelectableTranscript(lambda: fragments, lambda: None, lambda text: None)

        def check(width=80):
            content = self.render(control, width)
            expected = FormattedTextControl(control.formatted_text).create_content(width, 20)
            self.assertEqual(content.line_count, expected.line_count)
            self.assertEqual(content.cursor_position, expected.cursor_position)
            self.assertEqual(content.menu_position, expected.menu_position)
            for i in range(content.line_count):
                self.assertEqual(content.get_line(i), expected.get_line(i))
                self.assertEqual(content.get_height_for_line(i, width, None),
                                 expected.get_height_for_line(i, width, None))
            return content

        first = check()
        check(3)
        fragments.append(('bold', 'added\n'))
        check()
        # Activity updates can replace text without changing the fragment count.
        fragments[-1] = ('bold', 'updated\nwith another line\n')
        self.assertEqual(check().line_count, 5)
        control.mouse_handler(mouse(MouseEventType.MOUSE_DOWN, 1, 0))
        control.mouse_handler(mouse(MouseEventType.MOUSE_MOVE, 3, 1))
        selected = check()
        self.assertTrue(any('selected-text' in style
                            for style, text in selected.get_line(0)))
        control.anchor = control.end = None
        check()
        replacement = []
        fragments[0] = ('class:link', 'hello', replacement.append)
        check()
        control.mouse_handler(mouse(MouseEventType.MOUSE_UP, 2, 0))
        self.assertEqual(len(replacement), 1)
        self.assertEqual(clicked, [])
        fragments.clear()
        self.assertEqual(check().line_count, 1)
        self.assertEqual(first.line_count, 3)

    def test_cursor_and_menu_markers_are_preserved(self):
        fragments = [('', 'hello\n'), ('[SetCursorPosition]', ''),
                     ('', 'world'), ('[SetMenuPosition]', '')]
        control = SelectableTranscript(lambda: fragments, lambda: None, lambda text: None)
        for _ in range(2):
            content = self.render(control)
            self.assertEqual(content.cursor_position, Point(0, 1))
            self.assertEqual(content.menu_position, Point(5, 1))


class SelectionTests(unittest.TestCase):
    def test_wheel_scrolls_even_over_a_link_without_changing_selection(self):
        clicked, scrolled = [], []
        control = SelectableTranscript(
            lambda: [('', 'link', clicked.append)], lambda: None, lambda text: None,
            scroll=scrolled.append,
        )
        control.create_content(80, 20)
        for kind in (MouseEventType.SCROLL_UP, MouseEventType.SCROLL_DOWN):
            self.assertIsNone(control.mouse_handler(mouse(kind, 1, 0)))
        self.assertEqual(scrolled, [-3, 3])
        self.assertEqual(clicked, [])
        self.assertIsNone(control.anchor)

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
