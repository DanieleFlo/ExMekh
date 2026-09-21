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
streaming and non-streaming calls, nested skill paths, saved JSON, calling-turn
thinking, and node isolation. The ordinary-tool cases require non-empty thinking
from the backend. See [Tool output caching](09-tool-call-cache.md) for the record
contract and offline checks.
