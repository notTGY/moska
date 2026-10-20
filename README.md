# moska

Control multiple agents from one terminal orchestrator.

Build with [uv](https://docs.astral.sh/uv/):

```sh
uv build
```

Share the wheel in `dist/`. Users with uv installed can install and launch it:

```sh
uv tool install /path/to/moska-0.1.0-py3-none-any.whl
moska
```

Requires Python 3.13+; uv can install it automatically. Run `uv tool update-shell` if `moska` is not on your PATH.

Moska's reports in the terminal use Markdown-aware styling for `#` headings,
bold, emphasis, inline code, fenced or indented code, and block quotes. HTTP(S)
links are clickable, including `[label](https://example.com)` links. Code blocks
keep their indentation and literal contents, including URLs.

Markdown source characters remain visible and are included when selecting and
copying text. Lists, tables, raw HTML, and other syntax remain readable as source;
HTML and images are not rendered. User and system messages remain literal text
with clickable URLs. Session history stores the original report text and reapplies
styling when restored.

Run the tests from the repository root:

```sh
PYTHONPATH=src uv run python -m unittest discover -s tests -v
```
