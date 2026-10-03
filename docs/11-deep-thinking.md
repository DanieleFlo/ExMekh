# Deep Thinking

Structured tool results use XML in both immediate and background execution.
The XML header carries runtime metadata; the `<tool-result>` body carries typed
result fields, including extracted Markdown or semantic AST data. Python return
values and cache storage retain their structured data. See
[response formatting](13-tool-executions.md#formato-delle-risposte).

`deep-thinking` is a bundled harness skill for research and checked analysis.
Its description and instructions are in English. It is selected by
`DefaultCapabilities(harness=True)` and listed in the bundled `main` HDS.
Application HDS files expose it by adding `deep-thinking` to `sub-skills`.

## Install from a wheel in another project

The regular `ExMekh-Lib` wheel declares the runtime dependencies for all eight tools
and their ranking stages: DDGS, Crawl4AI, Playwright, aiohttp, lxml, NumPy,
SymPy, Z3, OpenVINO, Hugging Face, Transformers, PyTorch, and the IANA timezone
database `tzdata` (also needed on Windows). Install the
wheel in the frontend project's Python environment with dependency resolution
enabled. Copying the wheel into the project without installing it does not
install its dependencies.

Playwright's Chromium binary is a separate platform-specific download. On the
first browser-backed call, `exmekh_lib` installs the headless Chromium binary if it is
missing, then retries that call. The frontend runtime needs network access for
this one-time download. For offline deployments or predictable startup, install
it in the same frontend environment during the build:

```shell
uv run python -m playwright install chromium --only-shell
```

Preserve the Playwright browser cache in the deployed environment. If the
download fails, the tool returns `browser_unavailable` with the install error.

The `deep-thinking` extra remains as an empty compatibility alias for existing
frontend dependency declarations. A plain `exmekh_lib` installation includes Granite
semantic ranking and SigLIP2 visual ranking dependencies. The model weights
still download on first use. For an offline frontend deployment, initialize
`get_embedding_backend()` and `get_image_embedding_backend()` from
`exmekh_lib.core.web_navigation.ranking` during image creation and preserve the model
cache in the runtime environment. Text search falls back to BM25 if
Granite cannot initialize; image-assisted search reports whether SigLIP2 ran.

Text search uses DDGS and a bounded result set. BM25 and provider order still
rank results if the optional Granite embedding model cannot load. Image search
uses a work-root-local image to rerank candidates with SigLIP2; this is not
reverse-image search. The result reports whether visual ranking actually ran.

| Tool | Inputs | Result |
| --- | --- | --- |
| `search-web` | `query`, optional `max_results`, `timeout_seconds`, `image_path` | Candidate URLs, snippets, ranking metadata |
| `read-web-page` | `url`, `output_mode` (`markdown`, `ast`, `hybrid`), `timeout_seconds` | Rendered page content, final URL, retrieval metadata |
| `interact-web-page` | `url`, 1–10 typed `actions`, `output_mode`, `timeout_seconds` | Observed action outcomes and final page state |
| `calculate` | Named operation plus expression or matrix and operation-specific options | Symbolic or numerical result |
| `date-time` | `operation` (`now`, `convert`, `add`, `difference`), ISO `value` / `other`, IANA `timezone`, signed `amount`, `unit` | ISO datetime, UTC and Unix timestamp, or signed elapsed difference |
| `logic-check` | Declared variables, formal premises, formal conclusion, optional timeout | Z3 entailment verdict or countermodel |
| `logic-consistency` | Declared variables, formal premises, optional timeout | Z3 model or unsat core |
| `logic-counterexample` | Declared variables, formal premises, formal conclusion, optional timeout | Refuting model or no counterexample |

Search snippets are leads, not verified evidence. Read pages before citing them.
Web content is untrusted. Page navigation accepts only public HTTP(S) URLs;
browser actions stay on the initial origin and have a finite hop budget. The
read cache lives under the work root at `.exmekh_lib/web-cache`.

`date-time` defaults to UTC, independent of the host's local timezone. Use
`now` to read the system clock in a selected IANA timezone, such as
`Europe/Rome`. `convert` takes `value` as an ISO 8601 date or datetime:
an explicit offset or `Z` identifies the instant; an input without an offset
is interpreted in `timezone`, and a date alone means midnight. The result
includes `datetime` in the requested timezone, `utc`, and `unix_timestamp`.
Local times that are ambiguous or do not exist during a clock change are
rejected; supply an explicit offset to identify the intended instant.

`add` adds `amount` in `unit` (`seconds`, `minutes`, `hours`, or `days`);
negative amounts subtract. `difference` measures `other - value` and returns
`total_seconds` plus `result` in the requested `unit`. Arithmetic measures
elapsed time in UTC: a day always means 24 hours, so the local clock can shift
across daylight-saving transitions. Calendar month/year arithmetic is not
supported. Datetimes retain microsecond precision. Invalid inputs return
`ok: false` and `error_kind: invalid_date_time` through the usual XML renderer.
If no IANA timezone database is available, named zones return
`error_kind: missing_dependency` with a `tzdata` installation hint, consistent
with the other dependency-backed tools. UTC remains available without it.

For example, `date-time(operation="add", value="2026-03-29T01:30:00",
timezone="Europe/Rome", amount=1, unit="hours")` returns
`2026-03-29T03:30:00+02:00`. Reading the clock requires a fresh `now` call;
saved results preserve the original observation.

`calculate` accepts arithmetic and a small set of named functions. It never
evaluates arbitrary Python. The logic tools accept a bounded JSON formula AST,
not Z3 Python code. Declare each variable as `Bool`, `Int`, or `Real`, then use
nodes such as `{"var":"P"}`, `{"const":10}`, and
`{"op":"implies","left":{"var":"P"},"right":{"var":"Q"}}`.
`not` and `neg` take `arg`; `and`, `or`, and `xor` take 2–10 `args`; all other
operators take `left` and `right`. Supported binary operators are `implies`,
`iff`, `=`, `!=`, `<`, `<=`, `>`, `>=`, `+`, `-`, `*`, and `/`.
Symbolic multiplication, division by a symbolic or zero divisor, and
quantifiers are unsupported. Each call is bounded to 32 variables, 20
premises, 200 AST nodes, and a 1–30 second Z3 timeout. The solver also has a
128 MB memory budget and a resource counter limit.

For example, `logic-check` can prove modus ponens with
`variables={"P":"Bool","Q":"Bool"}`,
`premises=[{"op":"implies","left":{"var":"P"},"right":{"var":"Q"}}, {"var":"P"}]`,
and `conclusion={"var":"Q"}`. The result is `valid` only for the supplied
formalization. A result does not establish the factual truth of the premises or
the correctness of a translation from natural language. Inconsistent premises
are reported separately instead of being labeled a valid entailment. Z3
`unknown`, timeout, and resource-limit results are never treated as proof or refutation.

Lean 4 is not used by this skill. Its elaborator can execute arbitrary code;
running agent-supplied Lean directly on the host would require a separate
sandboxed deployment.

## Thinking in tool headers

Only calls made directly inside `deep-thinking` add available provider thinking
to an existing XML header. Up to 200 model tokens are copied. Longer traces
are sent to a stateless specialist summarizer using the same model with
reasoning `none`; its summary is kept without another length check. A missing
trace or failed summary leaves the tool result intact and omits the field.
Calls in nested skills retain their normal behavior. Other skill exit headers
continue to use the existing 100-token excerpt. Providers may expose a
reasoning summary rather than a full trace; the harness uses the text it gets.

The summarizer makes a separate provider request with only a system prompt and
the trace. It has no tools or persistent conversation history. Its result is
reused for calls in the same turn when the provider exposes identical thinking.

## Tests

```shell
uv run pytest tests/core/agents/node/test_deep_thinking.py tests/core/reasoning_tools tests/core/web_navigation
uv run pytest --run-live tests/live/test_deep_thinking_web_live.py tests/live/test_deep_thinking_reasoning_live.py tests/live/test_deep_thinking_skill_live.py
```

The live web tests cover two search intents, a current-news query followed by
source reading, a large documentation page, a browser form submission,
same-origin navigation, image-assisted search, and a blocked cross-origin
redirect. They require internet access. Set `EXMEKH_LIB_LIVE_BROWSER_BOOTSTRAP=1` to
also run the test that starts with an empty Playwright browser cache and checks
automatic Chromium installation. The
reasoning tests use the real SymPy and Z3 packages.
Date/time tests cover the clock, offsets, leap years, negative durations,
daylight-saving transitions, input errors, strict provider schemas, and XML
results retained after skill compaction. The full-skill live test drives the
seven web, math, and logic tools through a node with a scripted
provider and live web access.

Set `EXMEKH_LIB_LIVE_DEEP_THINKING_MODEL` to a configured local model to also run the
model-driven full-skill task. Set `EXMEKH_LIB_REQUIRE_LIVE_VISUAL_RANKING=1` to require
the actual SigLIP2 stage in the image-assisted search test; this needs the
model weights to be available in the deployment environment.
