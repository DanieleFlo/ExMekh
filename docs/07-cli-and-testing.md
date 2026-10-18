# CLI, Testing & Live Verification

This document covers running the interactive command-line interface, executing standard offline unit tests, and running live integration tests against model backends.

For a separately invoked, scored comparison of tool argument compilation with
full schemas versus scoped parameter documentation, see
[Tool argument evaluations](../evaluations/README.md). Its 14 paired scenarios
are independent of the live-test suite and do not execute tools.

---

## 1. Command-Line Interface (CLI)

The `exmekh_lib` package exposes an interactive CLI script defined in [`exmekh_lib.main:main`](../src/exmekh_lib/main.py).

### Launching the CLI
Run the application using `uv`:

```shell
uv run exmekh-lib
```

Alternatively, run the main module directly:

```shell
python -m exmekh_lib.main
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

### After moving or renaming the checkout

Virtual environment activation scripts and Python entry points contain absolute
paths. After moving or renaming the checkout, regenerate the
activation scripts and reinstall the locked packages so commands no longer
reference the old directory:

```shell
uv venv --allow-existing .venv
uv sync --locked --reinstall
```

The first command preserves installed packages. The second regenerates their
entry points without changing `uv.lock`. While repairing the environment,
`uv run --locked python -m pytest` can run tests without the stale `pytest` script.

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

Pytest uses `pytest-xdist` to run tests in parallel on all logical CPUs available
to its process. `PYTEST_XDIST_AUTO_NUM_WORKERS` overrides this automatic count.
WSL CPU limits and process affinity apply to automatic detection.
Tests marked `live` or `serial` share one worker and run sequentially with each
other; the remaining tests can run concurrently. Live tests still require opt-in.

Use `@pytest.mark.serial` on a test, or `pytestmark = pytest.mark.serial` on a
module, when its shared resources make it incompatible with parallel execution.
Keep `--dist=loadgroup` when changing the worker count so this grouping is preserved:

```shell
uv run pytest -n 16
uv run pytest -n 0
PYTEST_XDIST_AUTO_NUM_WORKERS=16 uv run pytest
```

The first command uses sixteen workers; the second runs the entire suite
serially. The environment variable sets the count used by `-n auto`. Choosing
more workers than the available CPUs does not increase CPU capacity.

To exclude live tests even when the environment enables them:

```shell
uv run pytest -m "not live"
```

For the tool output cache and runtime integration:

```shell
uv run pytest tests/core/cache tests/core/tool_calls tests/core/agents/node tests/core/tokenizers -q
```

If pytest cannot access the default temporary directory on Windows, select a
fresh workspace directory with `--basetemp=.exmekh_lib/pytest-doc-check`. Pytest clears
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
uv run pytest --cov=exmekh_lib
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
EXMEKH_LIB_RUN_LIVE_TESTS=1 uv run pytest tests/live
```

### Environment Configuration for Live Tests
Live tests read endpoint settings from `.env`:

```ini
BASE_URL=http://localhost:8080/v1
API_KEY=sk-local-key
EXMEKH_LIB_LIVE_MODEL=qwen3.5-9b-mtp
```

### Assertions and coverage

Live tests exercise the software through real provider responses. Compare actual
tool arguments, receipts, histories and effects before comparing an answer with
the requested fixture. A plausible model answer cannot replace these checks, and
different model choices cannot excuse effects that disagree with emitted arguments.

`CLIENT`, `PROVIDER`, `IMPLEMENTATION`, `CACHE` and `PROCESS` assertions identify
the contract being checked. `COVERAGE` means the required branch was not exercised;
`FLOW` means the workflow did not terminate as required. Neither is converted to
a passing result or a model-limitation skip. Inspect the recorded requests and
responses before attributing a failure to either the model or the implementation.
Environment and fixture requirements may still produce explicit skips.

The file-editing observer checks literal uniqueness, physical line boundaries,
disjoint edits and resulting bytes independently of the production selection
implementation. Offline tests deliberately supply inconsistent successful receipts
to verify that this observer catches bugs.

File-operation live cases expose one real tool at a time and dispatch the actual
returned call through its hooks and the real file runtime. Edit prerequisites are
real inspection/read results in a valid history; snapshot authorization still
crosses the live interactive protocol. These cases do not depend on entering,
leaving or reopening a skill. Neither responses nor arguments are substituted,
and the recorder performs no additional inference to complete a model workflow.
The requested software branch must still be exercised, such as an empty
replacement or a call with two edits. Bytes are compared with the actual
arguments, rather than treating a different model-selected replacement as a
filesystem bug. Removing an empty replacement must preserve any unselected
trailing newline; it does not necessarily produce a zero-byte file.

Reading live cases use inspection and reading directly at root. They assert the
selected text or PDF pages, actual image companions delivered to the provider,
and identical values and images recovered from the cache. An optional image
inspection is allowed; image reads do not require it. These checks do not depend
on a model's final wording or skill-entry strategy. Mixed-content tests retain
their separate end-to-end checks that the endpoint consumes the supplied images.

The public-web suite also verifies actual 401/403, 404, 429 and 503 responses:
the error family, cause, status, requested URL and recovery guidance must agree.

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
60-second default wait. Set `EXMEKH_LIB_LIVE_REQUEST_TIMEOUT_SECONDS` to a finite positive
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
suite remains live, and the complete real-provider deep-thinking audit still
requires `EXMEKH_LIB_LIVE_DEEP_THINKING_MODEL`.

The audit runs a declared sequence of real calls. Its provider fixture keeps the
tool registry unchanged and narrows only the allowed names for each step; changing
registered definitions inside one DS4 context produces HTTP 409. Required inputs
and optional defaults come from the step and the tool's public signature.
Responses and returned arguments are preserved. Work runs synchronously in this
fixture, while the separate process suite verifies managed and background work.
After hooks observe unmodified results before presentation, so a recorded tool
name cannot hide an error result. The audit checks the actual mean, logic statuses,
HTTP response and navigation effect. Child calls are inspected through
`skill_executions`; the root's exported history contains the root calls.

Public search discovery and reading share one live check: results must be
bounded, unique public URLs, and at least one discovered source must be readable.
The check does not require a particular domain to appear among the first five
results of a changing search index. Exhausted provider failures remain failures.

### Mixed user-content live checks

```shell
BASE_URL=http://127.0.0.1:8081/v1 \
EXMEKH_LIB_LIVE_IMAGE_PATH=/path/to/earth.jpg \
uv run pytest -n 0 --run-live tests/live/test_mixed_user_inputs_live.py -q
```

Use an image-capable server with its matching encoder. The tests send ordered
text/image lists through both APIs, check retained colors and a text reference
across turns, and inspect a real Earth photo. Reasoning is disabled to keep the
output budget focused on the asserted reply. The photo case needs the provided
fixture path.

Native file tests compare the original single-file input with file/text and
text/image/file/text lists. They require reading a reference present only in the
attachment bytes; accepting a request while ignoring the attachment fails the
test. These opt-in checks expose endpoint limitations without changing exmekh_lib's
native file transport.

Local verification on 2026-10-02 used Qwen3.8-27B-UD-Q4_K_S with its matching
F16 vision encoder, an 8192-token context, and reasoning disabled. All eight
text/image and real-photo cases passed across Chat Completions and Responses.
All six native TXT-file cases failed, including the unchanged single-file
input: Chat Completions returned `UNREADABLE`, and Responses rejected
`input_file` with HTTP 400 (`invalid JSON request`). The DS4 content parsers
accept text and images but do not ingest native file attachments. This is an
endpoint limitation already present with the original file message, rather
than a regression caused by mixed content. The native-file checks remain
strict and opt-in, so the complete live command currently reports these six
failures on this engine.

The local JUnit report is `.exmekh_lib/mixed-input-live-20261002/nonthinking-results.xml`;
the accompanying `server.log` retains the inference log. These generated
artifacts are ignored by Git. The dedicated test server was stopped after
verification.

The subsequent refactoring passed the full offline suite, 100% coverage,
Ruff and Pyright. SDK payload tests verify the final ordered content and file
indices. Live checks could not be repeated at final review: DS4 was already
running on port 8080 without a vision encoder, and rejected a second server.
That existing process was left running; the live results above describe the
earlier verification.

### Verification after the package rename (2026-10-02)

The full live run verified agent scopes, strict tool arguments, mixed text/image
history, sub-agent memory, saved tool outputs, and ordinary and streaming child
process execution. The browser form fixture now uses `https://httpbin.org/forms/post`
and checks that the submitted fields appear in the captured destination page;
the previous Heroku fixture timed out even with standalone Chromium.
All seven public-web checks passed, including browser provisioning from an empty
cache. Both rejection-recovery checks passed on the configured provider, including
mixed image/file rejection and a subsequent successful image request.

Across these runs, 29 live cases passed. The six native TXT-file cases still fail
with HTTP 400: the configured DS4 endpoint explicitly supports only text and
images. Native transport and strict file-reading assertions are preserved. The
full model-driven deep-thinking audit and two real-photo cases were not run
because their optional model and photo-fixture settings were absent.

The offline suite passed all 2,024 cases with 100% branch coverage. Pyright,
Ruff, formatting, and wheel/source builds passed. Local reports are stored under
`.exmekh_lib/live-verification-20261002/` and are ignored by Git.
