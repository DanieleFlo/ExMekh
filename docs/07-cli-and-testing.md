# CLI, Testing & Live Verification

This document covers running the interactive command-line interface, executing standard offline unit tests, and running live integration tests against model backends.

---

## 1. Command-Line Interface (CLI)

The `qs` package exposes an interactive CLI script defined in [`qs.main:main`](file:///c:/Users/danie/Desktop/qs/qs/src/qs/main.py).

### Launching the CLI
Run the application using `uv`:

```shell
uv run qs
```

Alternatively, run the main module directly:

```shell
python -m qs.main
```

### CLI Features & REPL Loop
- **Streaming Printer**: Formats reasoning content (`Thinking: ...`) separately from final model output (`Response: ...`).
- **Execution Event Logging**: Logs tool invocations with depth and skill path information:
  ```
  tool_start: mock-tool (depth=0, path=/)
  ```
- **Session History Dumps**: Prints JSON representations of both linear conversation history and hierarchical `Deep skill history` after each step.
- **Session Exit**: Type `exit` or `quit` to exit the loop.

---

## 2. Running Offline Unit Tests

Unit tests verify client models, capabilities loading, schema coercion, node execution, and provider mappers without contacting external LLM servers.

### Static checks

```shell
uv run --locked pyright
uv run --locked ruff check src tests
uv run --locked ruff format --check src tests
```

Pylance and the Pyright CLI share the `[tool.pyright]` configuration in
`pyproject.toml`. The type check covers all production code under `src`, targets
Python 3.13, and uses the project's `.venv`. VS Code selects that environment
by default and reports diagnostics across the workspace.

The project uses `standard` checking with strict inference for lists,
dictionaries and sets. Missing generic type arguments, uninitialized instance
attributes, unnecessary casts and unused `type: ignore` comments are errors.
Fix the underlying types before adding an exception. Test code remains covered
by Ruff and pytest; the Pyright CLI gate covers `src`.

See the [Pyright configuration reference](https://github.com/microsoft/pyright/blob/main/docs/configuration.md)
for the shared settings.

### Executing Unit Tests

```shell
uv run pytest
```

To exclude live tests even when the environment enables them:

```shell
uv run pytest -m "not live"
```

For the tool output cache and runtime integration:

```shell
uv run pytest tests/core/cache tests/core/tool_calls tests/core/agents/node tests/core/tokenizers -q
```

If pytest cannot access the default temporary directory on Windows, select a
fresh workspace directory with `--basetemp=.qs/pytest-doc-check`. Pytest clears
the selected directory when starting: use a dedicated disposable path.

### Coverage Requirements
The project configuration in `pyproject.toml` mandates 100% test coverage for core components:

```toml
[tool.coverage.report]
fail_under = 100
show_missing = true
```

Run pytest with coverage reporting:

```shell
uv run pytest --cov=qs
```

---

## 3. Live Model Integration Testing

Live integration tests in `tests/live/` interact with real LLM provider endpoints to verify tool calling, multi-step navigation, and model instruction adherence.

### Test Opt-In Methods
By default, `uv run pytest` skips `@pytest.mark.live` tests to avoid external API calls. You can opt in using either method below:

#### Method A: Pytest Command Flag

```shell
uv run pytest --run-live tests/live
```

#### Method B: Environment Variable

```shell
QS_RUN_LIVE_TESTS=1 uv run pytest tests/live
```

### Environment Configuration for Live Tests
Live tests read endpoint settings from `.env`:

```ini
BASE_URL=http://localhost:8080/v1
API_KEY=sk-local-key
QS_LIVE_MODEL=qwen3.5-9b-mtp
```

### Assertion Diagnostic Conventions
Assertion messages in live tests use prefix tags to clarify failure causes:
- **`CLIENT`**: Protocol error, history invariant failure, or client library bug.
- **`MODEL`**: Model failed to follow prompt instructions, output invalid tool calls, or generated unexpected text.

### Tool cache live checks

```shell
uv run pytest --run-live tests/live/test_tool_cache_live.py -v
```

These tests use the Responses API and a local test vocabulary. They exercise
streaming and non-streaming calls, saved JSON, calling-turn thinking when supplied
by the backend, and model-driven search and reading of saved skill output.
Cache isolation, nested paths, exit retry and instruction delivery are covered
deterministically in `tests/core/agents/node/`. See
[Tool output caching](09-tool-call-cache.md) for the record contract.

### Live tool execution checks

```shell
uv run pytest --run-live tests/live/test_tool_executions_live.py -v -rs
```

This suite uses the real Responses provider and real Python child processes.
It checks ordinary and streaming requests, fast completion with default execution
settings, background waits and automatic final delivery, and incremental stdout
with output cursors. Prompts request only a minimal final reply.

All model live tests use a 900-second HTTP timeout, independently of the tool's
60-second default wait. Set `QS_LIVE_REQUEST_TIMEOUT_SECONDS` to a finite positive
number to override the HTTP timeout. Automatic request retries are disabled so a
failed stateful request is not repeated silently.

The fixture releases externally gated work at observed runtime events. Model
responses and tool calls are never replaced. Each run has an isolated agent
context, so a backend with persistent tool registries cannot mix test cases.
Processes are stopped and pipes closed during cleanup, including on failure.

Assertions check runtime invariants and whether actual output reached the model.
Additional model attempts are allowed; required calls and progress must still be
present. IDs, cursor advancement, one stored record and one automatic delivery
per output remain strict invariants. `CLIENT`, `CACHE` and `PROCESS` failures
indicate protocol/runtime, persistence and subprocess fixture problems respectively.
Provider HTTP errors and timeouts remain visible failures and are not converted
into passing checks or missing-coverage skips.
Each fixture writes `live-trace.json` under pytest's temporary directory with real
calls, output headers, scope paths and call durations for diagnosis. It contains
only generated test data and does not include provider credentials.

The corresponding offline checks use a scripted provider with real runtime and
subprocesses. Overlapping live cases for hooks, cancellation, nonzero exits,
scope delivery, reentry, concurrency and uncached results are consolidated here:

```shell
uv run pytest tests/core/agents/node tests/core/tool_calls/test_execution_manager.py tests/core/tool_calls/test_live_execution_support.py -q
```

The default minute deadline is checked with a simulated wait expiry and a real
gated process; the offline test verifies the 60-second budget without sleeping
for a minute. Manual final reads are checked both individually and in the same
batch as an execution-control wait.

The deep-thinking workflow uses a scripted provider and mocked web boundaries in
`tests/core/agents/node/test_deep_thinking_workflow.py`. Clock, SymPy and Z3
capability integration checks run offline in
`tests/core/agents/capabilities/test_reasoning_integration.py`. The real public-web
suite remains live, and the complete model-driven deep-thinking audit still
requires `QS_LIVE_DEEP_THINKING_MODEL`.

Public search discovery and reading share one live check: results must be
bounded, unique public URLs, and at least one discovered source must be readable.
The check does not require a particular domain to appear among the first five
results of a changing search index. Exhausted provider failures remain failures.
