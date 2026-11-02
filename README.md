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

Use `/new` to save the current memory and start a blank session. This stops active
work and clears the chat and model context while keeping your model and agent
settings. Previous conversations and checkpoints remain in the workspace's session
database; reopening Moska resumes the new session. `moska --export-session FILE`
exports the saved conversation history as JSON. `/clear` only clears the visible chat.
