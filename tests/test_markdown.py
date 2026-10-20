import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage
from prompt_toolkit.data_structures import Point
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.output import DummyOutput

from orchestrator.storage import SessionStore
from tui.app import ChatUI
from tui.markdown import markdown_fragments
from tui.transcript import SelectableTranscript


REPORT = '''# Results

**Passed** with *notes* and `literal **code**`.

- first
  - nested
1. ordered
> A qualification

```python
    print("**literal** https://example.com/code")
```

| Check | Result |
| --- | --- |
| Test | Passed |

[Evidence](https://example.com/report)
'''


class MarkdownTests(unittest.TestCase):
    def test_styles_common_markdown_and_preserves_every_character(self):
        fragments = markdown_fragments(REPORT)
        self.assertEqual(''.join(item[1] for item in fragments), REPORT)
        for style, text in (
            ('markdown-heading', '# Results\n'),
            ('markdown-strong', '**Passed**'), ('markdown-emphasis', '*notes*'),
            ('markdown-code', '`literal **code**`'),
            ('markdown-quote', '> A qualification\n'),
        ):
            self.assertTrue(any(style in item[0] and item[1] == text for item in fragments))
        code = next(item for item in fragments if 'print(' in item[1])
        self.assertEqual(code[0], 'class:markdown-code')
        self.assertEqual(len(code), 2)  # URLs inside code are literal.

    def test_incomplete_and_unsupported_markdown_remains_readable(self):
        for source in ('', '**unfinished', '```py\n  x = 1',
                       '~~~\n```\n**literal**\n~~~\n**bold**',
                       '\\*escaped* snake_case <b>HTML</b> ![image](file.png)',
                       '标题\n====\n\n| a | b |\n| - | - |',
                       '  text\n\n    indented **code**\n', 'no trailing newline'):
            with self.subTest(source=source):
                self.assertEqual(''.join(item[1] for item in markdown_fragments(source)), source)
        self.assertFalse(any('markdown-emphasis' in item[0]
                             for item in markdown_fragments('\\*escaped* snake_case')))

    def test_links_open_destination_and_never_execute_other_schemes(self):
        fragments = markdown_fragments(
            '[Evidence](https://example.com/report) https://example.com/plain '
            '[bad](javascript:alert(1)) [local](file:///tmp/report)')
        links = [item for item in fragments if len(item) == 3]
        self.assertEqual(len(links), 2)
        event = MouseEvent(Point(0, 0), MouseEventType.MOUSE_UP, MouseButton.LEFT, frozenset())
        with patch('tui.markdown.webbrowser.open') as opened:
            for item in links:
                item[2](event)
        self.assertEqual([call.args[0] for call in opened.call_args_list],
                         ['https://example.com/report', 'https://example.com/plain'])

    def test_link_destination_preserves_parentheses(self):
        fragments = markdown_fragments('[Details](https://example.com/entry_(version))')
        link = next(item for item in fragments if len(item) == 3)
        event = MouseEvent(Point(0, 0), MouseEventType.MOUSE_UP, MouseButton.LEFT, frozenset())
        with patch('tui.markdown.webbrowser.open') as opened:
            link[2](event)
        opened.assert_called_once_with('https://example.com/entry_(version)')

    def test_selection_copies_markdown_source_across_styled_fragments(self):
        copied = []
        control = SelectableTranscript(lambda: markdown_fragments(REPORT), lambda: None, copied.append)
        for kind, point in ((MouseEventType.MOUSE_DOWN, Point(0, 0)),
                            (MouseEventType.MOUSE_UP, Point(0, len(REPORT.splitlines())))):
            control.mouse_handler(MouseEvent(point, kind, MouseButton.LEFT, frozenset()))
        self.assertEqual(copied, [REPORT])
        self.assertTrue(any('selected-text' in item[0] for item in control.formatted_text()))


class MarkdownReportTests(unittest.IsolatedAsyncioTestCase):
    async def test_response_storage_and_restore_preserve_markdown(self):
        class Agent:
            async def astream(self, *args, **kwargs):
                yield {'model': {'messages': [AIMessage(content=REPORT)]}}

        with tempfile.TemporaryDirectory() as directory:
            session = SessionStore(directory, data_dir=directory).open()
            try:
                with patch('adapters.providers.registry.OpenAIConnection', return_value=SimpleNamespace(current_account=lambda: None)), create_pipe_input() as pipe:
                    ui = ChatUI(Agent(), session=session, input=pipe, output=DummyOutput())
                    ui.add('you', '**literal user input**')
                    await ui.respond('generate report')
                    text = ''.join(item[1] for item in ui.render_transcript())
                    self.assertIn(REPORT, text)
                    self.assertTrue(any('markdown-heading' in item[0] for item in ui.fragments))
                    self.assertIn(('', '**literal user input**'), ui.fragments)
                    stored = session.export()['transcript']
                    self.assertEqual(stored[-1]['text'], REPORT)
                    restored = ChatUI(Agent(), session=session, input=pipe, output=DummyOutput())
                    self.assertEqual(''.join(item[1] for item in restored.render_transcript()), text)
                    self.assertTrue(any('markdown-heading' in item[0] for item in restored.fragments))
            finally:
                session.close()
