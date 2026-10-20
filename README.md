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
