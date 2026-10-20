"""Source-preserving Markdown styling for prompt-toolkit reports.

Keep every character so selection, code indentation, tables and unsupported
Markdown remain usable without a separate source/display offset mapping.
"""

import re
import webbrowser

from prompt_toolkit.mouse_events import MouseEventType


INLINE = re.compile(
    r'(?P<code>`+)(?P<code_text>.*?)(?P=code)(?!`)|'
    r'(?P<link>\[[^\]\n]+\]\((?P<target>https?://(?:[^\s<>()]|\([^\s<>()]*\))+)\))|'
    r'(?P<url>https?://[^\s<>]+)|'
    r'(?P<strong>\*\*(?=\S).+?(?<=\S)\*\*|__(?=\S).+?(?<=\S)__)|'
    r'(?P<emphasis>(?<!\w)\*(?=\S)[^*\n]+?(?<=\S)\*(?!\w)|'
    r'(?<!\w)_(?=\S)[^_\n]+?(?<=\S)_(?!\w))'
)
FENCE = re.compile(r'^ {0,3}(`{3,}|~{3,})(.*)$')


def link_fragment(text, url, style=''):
    def open_link(event):
        if event.event_type == MouseEventType.MOUSE_UP:
            webbrowser.open(url)

    return (style + ' class:link' if style else 'class:link', text, open_link)


def plain_fragments(text):
    """Linkify literal user/system text without interpreting Markdown."""
    fragments, start = [], 0
    for match in re.finditer(r'https?://[^\s<>]+', text):
        fragments.append(('', text[start:match.start()]))
        fragments.append(link_fragment(match[0], match[0]))
        start = match.end()
    fragments.append(('', text[start:]))
    return fragments


def markdown_fragments(text):
    """Style common Markdown while preserving its exact source, even if incomplete."""
    fragments = []
    fence = None
    for line in text.splitlines(keepends=True):
        marker = FENCE.match(line)
        if fence:
            fragments.append(('class:markdown-code', line))
            if (marker and marker[1][0] == fence[0]
                    and len(marker[1]) >= len(fence) and not marker[2].strip()):
                fence = None
            continue
        if marker:
            fence = marker[1]
            fragments.append(('class:markdown-code', line))
            continue
        if line.startswith(('    ', '\t')):
            fragments.append(('class:markdown-code', line))
            continue
        style = ''
        if re.match(r'^ {0,3}#{1,6}(?:\s|$)', line):
            style = 'class:markdown-heading'
        elif re.match(r'^ {0,3}>', line):
            style = 'class:markdown-quote'
        start = 0
        for match in INLINE.finditer(line):
            # Backslash-escaped Markdown punctuation is literal.
            if (len(line[:match.start()]) - len(line[:match.start()].rstrip('\\'))) % 2:
                continue
            fragments.append((style, line[start:match.start()]))
            if match['code']:
                fragments.append((style + ' class:markdown-code', match[0]))
            elif match['link'] or match['url']:
                fragments.append(link_fragment(match[0], match['target'] or match['url'], style))
            else:
                fragments.append((style + (' bold' if match['strong'] else ' italic'), match[0]))
            start = match.end()
        fragments.append((style, line[start:]))
    return fragments
