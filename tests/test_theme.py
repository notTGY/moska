import unittest
from types import SimpleNamespace
from unittest.mock import patch

from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from tui.app import ChatUI
from tui.markdown import markdown_fragments


def contrast(foreground, background):
    def luminance(color):
        channels = [int(color[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        linear = [c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4
                  for c in channels]
        return sum(c * weight for c, weight in zip(linear, (.2126, .7152, .0722)))

    light, dark = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (light + .05) / (dark + .05)


class ThemeTests(unittest.TestCase):
    def setUp(self):
        connection = SimpleNamespace(current_account=lambda: None)
        with patch('adapters.providers.registry.OpenAIConnection', return_value=connection), create_pipe_input() as pipe:
            self.ui = ChatUI(None, input=pipe, output=DummyOutput())
        # Include toolkit defaults: these can override inherited colors or reverse them.
        self.style = self.ui.app._merged_style

    def attrs(self, classes):
        return self.style.get_attrs_for_style_str(classes)

    def test_airi_canvas_and_accents_are_applied_to_application(self):
        canvas = self.attrs('')
        self.assertEqual((canvas.color, canvas.bgcolor), ('ffffff', '161c27'))
        for role, color in (
            ('brand', '2fbead'), ('agent', '2fbead'), ('moska', '2fbead'),
            ('markdown-heading', '2fbead'), ('accent', '84e0d5'),
            ('muted', '909393'), ('markdown-quote', '909393'),
            ('link', '92c7ff'), ('system', 'ffffff'), ('markdown-code', '84e0d5'),
            ('markdown-strong', '3debcb'), ('markdown-emphasis', '84e0d5'),
        ):
            with self.subTest(role=role):
                self.assertEqual(self.attrs('class:' + role).color, color)
        self.assertTrue(self.attrs('class:brand').bold)
        self.assertTrue(self.attrs('class:link').underline)
        self.assertTrue(self.attrs('class:markdown-quote').italic)

    def test_interactive_surfaces_keep_readable_brand_colors(self):
        for classes, background in (
            ('class:composer', '24332f'),
            ('class:completion-menu.completion', '005a60'),
            ('class:completion-menu.meta.completion', '005a60'),
            ('class:completion-menu.completion.current', '066d74'),
            ('class:completion-menu.meta.completion.current', '066d74'),
            ('class:dialog,dialog.body', '161c27'),
            ('class:dialog,dialog.body class:radio-checked class:radio-selected', '066d74'),
            ('class:agent-option class:agent-option.focused', '066d74'),
            ('class:agent-option class:agent-option.focused class:agent-option.disabled', '161c27'),
            ('class:markdown-code class:selected-text', '005a60'),
            ('class:link class:selected-text', '005a60'),
            ('class:dialog,dialog.body class:frame.label', '005a60'),
            ('class:muted', '161c27'),
        ):
            with self.subTest(classes=classes):
                attrs = self.attrs(classes)
                self.assertEqual(attrs.bgcolor, background)
                self.assertFalse(attrs.reverse)
                self.assertGreaterEqual(contrast(attrs.color, attrs.bgcolor), 4.5)
        title = self.attrs('class:dialog,dialog.body class:frame.label')
        self.assertEqual(title.color, '3debcb')
        self.assertEqual(self.attrs('class:dialog class:frame.border').color, '2fbead')
        for classes in ('class:dialog class:shadow', 'class:dialog,dialog.body class:shadow'):
            self.assertEqual(self.attrs(classes).bgcolor, '161c27')
        disabled = self.attrs('class:agent-option.focused class:agent-option.disabled')
        self.assertTrue(disabled.italic)
        self.assertTrue(disabled.underline)

    def test_rendered_markdown_has_distinct_readable_styles(self):
        source = ('Plain **strong** *emphasis* `inline`\n'
                  '# Heading\n> Quote\n```python\nprint("code")\n```\n'
                  '    indented code\n')
        fragments = markdown_fragments(source)
        self.assertEqual(''.join(fragment[1] for fragment in fragments), source)
        for text, color, bold, italic in (
            ('Plain ', 'ffffff', False, False),
            ('**strong**', '3debcb', True, False),
            ('*emphasis*', '84e0d5', False, True),
            ('`inline`', '84e0d5', False, False),
            ('# Heading\n', '2fbead', True, False),
            ('> Quote\n', '909393', False, True),
            ('```python\n', '84e0d5', False, False),
            ('print("code")\n', '84e0d5', False, False),
            ('```\n', '84e0d5', False, False),
            ('    indented code\n', '84e0d5', False, False),
        ):
            with self.subTest(text=text):
                style = next(fragment[0] for fragment in fragments if fragment[1] == text)
                attrs = self.attrs(style)
                self.assertEqual((attrs.color, attrs.bold, attrs.italic), (color, bold, italic))
                self.assertGreaterEqual(contrast(attrs.color, attrs.bgcolor), 4.5)
                selected = self.attrs(style + ' class:selected-text')
                self.assertEqual((selected.color, selected.bgcolor), ('ffffff', '005a60'))
                self.assertFalse(selected.reverse)


if __name__ == '__main__':
    unittest.main()
