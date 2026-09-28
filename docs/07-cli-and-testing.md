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

### Live tool execution checks

```shell
uv run pytest --run-live tests/live/test_tool_executions_live.py -v -rs
```

This suite uses the real Responses provider and real Python child processes,
including file reads, incremental stdout, nonzero exits and termination. It checks
ordinary and streaming requests, the actual default 60-second deadline, repeated
bounded waits, background work, cancellation, output cursors, final delivery,
skill reentry, concurrent jobs, sub-agent isolation, hooks and uncached results
across persistent turns. The minute deadline case takes at least one minute.

HTTP requests keep the SDK's default timeout (currently 600 seconds for reads),
independently of the tool's 60-second wait. For this suite, set
`QS_LIVE_REQUEST_TIMEOUT_SECONDS` to a finite positive number to override the
HTTP timeout. Automatic request retries are disabled so a failed stateful request
is not repeated silently.

The fixture releases externally gated work at observed runtime events. Model
responses and tool calls are never replaced. Each run has an isolated agent
context, so a backend with persistent tool registries cannot mix test cases.
Processes are stopped and pipes closed during cleanup, including on failure.

Assertions check runtime invariants and whether actual output reached the model;
they do not match final prose, reasoning text or answer formatting. A model that
does not exercise a specifically requested case, such as two calls in a single
response, produces an explicit `MODEL` skip for that case. This is missing live
coverage, not a successful check. `CLIENT`, `CACHE` and `PROCESS` failures indicate
protocol/runtime, persistence and subprocess fixture problems respectively.
Provider HTTP errors and timeouts remain visible failures and are not converted
into passing checks or missing-coverage skips.
Each fixture writes `live-trace.json` under pytest's temporary directory with real
calls, output headers, scope paths and call durations for diagnosis. It contains
only generated test data and does not include provider credentials.

The corresponding offline checks need no model endpoint:

```shell
uv run pytest tests/core/tool_calls/test_execution_manager.py tests/core/tool_calls/test_live_execution_support.py -q
```
