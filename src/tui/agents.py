"""Agent selector with unavailable entries and mouse/keyboard setup hints."""

from prompt_toolkit.data_structures import Point
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import FormattedTextControl, HSplit, Window
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.widgets import Label


class AgentSelector:
    def __init__(self, entries, enabled, accept, changed):
        self.entries = entries
        self.selected = {
            entry.spec.agent_id for entry in entries
            if entry.available and entry.spec.agent_id in enabled
        }
        self.focused = 0
        self.hint_index = 0
        self.changed = changed
        keys = KeyBindings()

        @keys.add('up')
        def up(event):
            self.focus(-1)

        @keys.add('down')
        def down(event):
            self.focus(1)

        @keys.add(' ')
        def toggle(event):
            self.toggle(self.focused)

        @keys.add('enter')
        def save(event):
            accept(set(self.selected))

        self.control = FormattedTextControl(
            self.fragments, focusable=True, key_bindings=keys,
            get_cursor_position=lambda: Point(x=0, y=self.focused), show_cursor=False,
        )
        self.container = HSplit([
            Label('↑↓ browse · Space toggle · Enter save · Esc cancel\n'),
            Window(self.control, height=max(1, len(entries)), wrap_lines=False),
            Window(height=1),
            Label(lambda: self.hint),
        ])

    @property
    def hint(self):
        return self.entries[self.hint_index].hint if self.entries else 'No agent types registered.'

    def focus(self, offset):
        self.focused = max(0, min(len(self.entries) - 1, self.focused + offset))
        self.hint_index = self.focused
        self.changed()

    def toggle(self, index):
        if not self.entries:
            return
        self.focused = self.hint_index = index
        entry = self.entries[index]
        if entry.available:
            agent_id = entry.spec.agent_id
            if agent_id in self.selected:
                self.selected.remove(agent_id)
            else:
                self.selected.add(agent_id)
        self.changed()

    def mouse(self, index, event):
        if event.event_type == MouseEventType.MOUSE_MOVE:
            self.hint_index = index
            self.changed()
        elif event.event_type == MouseEventType.MOUSE_UP:
            self.toggle(index)

    def fragments(self):
        fragments = []
        for index, entry in enumerate(self.entries):
            style = 'class:agent-option'
            if index == self.focused:
                style += ' class:agent-option.focused'
            if not entry.available:
                style += ' class:agent-option.disabled'
            checked = 'x' if entry.spec.agent_id in self.selected else ' '
            marker = f'[{checked}]' if entry.available else '[-]'
            status = 'installed' if entry.available else ('update required' if entry.path else 'not installed')
            if entry.version:
                status += f' · {entry.version}'

            def mouse(event, index=index):
                self.mouse(index, event)

            fragments.append((style, f' {marker} {entry.spec.name} · {status} ', mouse))
            fragments.append(('', '\n' if index < len(self.entries) - 1 else ''))
        return fragments

    def __pt_container__(self):
        return self.container
