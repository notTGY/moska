# moska

Control multiple Codex agents from one terminal orchestrator.

Install the core dependencies with `uv sync --locked` (Python 3.13+).

The retained `src/decomposer/` code is the orchestration starting point:

- `core.py`: the `new`, `run`, `fork`, and `wait` tools, agent/run state,
  result collection, and tool-call guards.
- `prompts.py`: orchestrator instructions, tool descriptions, and errors.
- `__init__.py`: lazy public import of `create_decomposer_agent`.
- `run_budget.py`: existing worker time-budget middleware.

The retained Python files are unchanged from the original code. The old model
adapters, vLLM launcher, and LangGraph development server launcher have been
removed. Dependencies cover only
imports used by the retained core; no model provider or inference engine is
installed.

The current core still communicates with LangGraph workers. Local Codex session
integration, the TUI, and automatic context compaction remain to be implemented.
Codex will use the existing on-device installation.
