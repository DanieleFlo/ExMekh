# Tool argument compilation evaluations

This standalone evaluation compares the representation before and after scoped
parameter documentation. It is outside `tests/live`, is never run by `pytest`,
and only contacts a model when its command is explicitly executed. The regular
suite tests the evaluator itself offline in `tests/evaluations`.

From the repository root, use the project's Python environment:

```sh
uv run python -m evaluations.tool_arguments --list
uv run python -m evaluations.tool_arguments --repetitions 3 --seed 42
```

For a first comparison with 28 requests:

```sh
uv run python -m evaluations.tool_arguments \
  --repetitions 1 --seed 20261018 \
  --output evaluations/results/tool-arguments-first-run.xml
```

The runner reads the repository `.env` and uses `LocalProvider` connection settings
and the configured model. It uses the Responses API. Override the model with
`--model`; use `--reasoning`, `--timeout` (seconds per request, default 120), and
`--max-output-tokens` when needed. Repeating `--case NAME` selects a subset.
The report path must end in `.xml`; the adjacent trace uses `.jsonl`.
Existing report/trace paths are rejected to avoid mixing runs.

## Paired conditions

- `full`: full parameter descriptions in the function schema; the skill catalog
  contains function descriptions only, matching the previous representation.
- `scoped`: structural function schemas without parameter descriptions; the
  production `SkillCatalogAttacher` generates parameter documentation in the
  active skill. Common execution instructions are supplied once in both modes.

Both conditions use the same capability policy as
`DeepAgent(parameter_descriptions_in_schema=...)`: `True` for `full` and `False`
for `scoped`.

The task, tool name, availability description, parameter constraints, model and
reasoning settings are identical within each pair. Every request has one available
tool and `tool_choice="required"`. There is no shared conversation history, skill
navigation or tool execution. These evaluations measure argument compilation after
the skill is open, not tool selection or end-to-end completion.

Each case is run in both modes consecutively, alternating which mode goes first.
Case order is shuffled reproducibly. `--seed` controls case order and generated
verification literals, not model sampling. Increasing `--repetitions` repeats all
pairs; it does not retry or repair failed answers.

## Scenarios

| Case | Capability |
| --- | --- |
| `customer_identifier` | Select the correct ID among distractors |
| `exclusive_date_boundary` | Translate inclusive dates to an exclusive bound |
| `money_in_minor_units` | Convert decimal currency amounts to cents |
| `nested_order_lines` | Compile ordered arrays of typed objects |
| `dataclass_postal_address` | Preserve accents and leading zeroes in nested data |
| `null_folder` | Distinguish null from literal text |
| `enum_direction_mapping` | Map user intent to documented enum values |
| `dictionary_description_key` | Read mapping-value docs and preserve a literal key |
| `union_opaque_code` | Choose the correct type within a union |
| `signed_coordinate_order` | Compile signed coordinates in documented order |
| `boolean_polarity_and_duration` | Handle boolean polarity and duration units |
| `literal_document_protocol` | Respect attribute escaping and literal body text |
| `documented_verification_literal` | Retrieve an exact value available only in docs |
| `background_execution` | Compile runtime execution options |

## Scoring

Each case has independently specified expected arguments. The grader reads the
raw SDK function-call argument string, before the library's value coercion.
Exactly one call to the designated tool is required; a wrong tool, multiple calls,
invalid JSON, duplicate keys or non-finite numbers score zero.

For a valid call, the score is the percentage of passed deterministic checks:
exact key sets at every object, lengths and order of arrays, and the expected
value at each leaf. Each case's score is on a 0–100 scale. The overall mean gives
each case equal weight when all requested repetitions complete. Fully correct
call rate is reported separately: it requires every check to pass. Checks for
common execution options contribute to the score along with business arguments.

Strings are exact, including whitespace in text protocols. Booleans cannot stand
in for numbers, and numeric strings are not converted to numbers. Numerically
equal integer representations are accepted; floating-point values use a 1e-9
tolerance. The only argument normalization is the runtime's documented equivalence
between null execution options and their defaults. Missing fields remain missing.

Infrastructure/API errors have no model score, remain visible in coverage counts,
and make the command exit with status 2. An incomplete model response scores zero.
The paired delta uses only pairs with a scored response on both sides. If there
are missing samples, inspect coverage and individual cases before comparing means.
Low model scores themselves do not fail the command: this is a measurement, not a
CI quality threshold.

The deterministic evaluator checks can be run without a model:

```sh
uv run pytest -n 0 tests/evaluations
uv run ruff check evaluations tests/evaluations
uv run pyright evaluations
```

## Reports and interpretation

The output XML contains run settings, planned/completed coverage, overall and
per-case scores, standard deviation, paired score delta, failed checks, expected
values, raw calls, latency and API usage. It is checkpointed after every request.
The adjacent JSONL file retains complete API requests and responses for diagnosis,
without credentials. XML uses the shared renderer; API traces intentionally keep
their original JSON protocol. Generated `evaluations/results/` files are ignored
by Git. Supply another `--output` path to retain a selected report elsewhere.

Tool-definition and instruction token counts use the project's Qwen tokenizer;
they are not the server's rendered prompt or a billing estimate. Actual API usage
is stored separately when the endpoint returns it. A short run on synthetic cases
does not prove quality equivalence: use repetitions, inspect exact-call rates and
expand the dataset with production failures.

## First recorded comparison

On 2026-10-18, `Qwen3.8-27B-UD-Q4_K_S` with reasoning `low`, seed `20261018`
and one repetition completed all 28 requests without infrastructure errors:

| Representation | Mean argument score | Fully correct calls |
| --- | ---: | ---: |
| Full schema descriptions | 97.80 / 100 | 13 / 14 |
| Scoped skill documentation | 96.02 / 100 | 12 / 14 |

The paired mean difference was -1.79 points for scoped documentation. Both modes
returned negative quantities and prices in `nested_order_lines`. In this run,
scoped documentation also produced XML entity text instead of the required
literal protocol in `literal_document_protocol`. The grader retained these
errors without repairing arguments or changing the expected answers.

This first report predates the catalog's switch to the shared XML renderer.
Its original requests and measurements are retained as historical evidence;
rerunning the command measures the current catalog presentation.

This is a preliminary observation from one sample per case, not a statistically
established difference. The reported standard deviation is across case samples,
not a confidence interval. The complete scored results are retained in the
[first-run XML report](../docs/test-evidence/tool-argument-compilation-2026-10-18.xml).

After the clean-code review and shared XML serialization, a second 28-request
run with the same model, seed and settings reproduced these aggregate scores,
exact-call counts and failing cases, with no infrastructure errors. Its
[reviewed-version XML report](../docs/test-evidence/tool-argument-compilation-reviewed-2026-10-18.xml)
retains the current presentation and measurements. Each run still uses only
one repetition per case.

The evaluation design follows [OpenAI's evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices):
task-specific checks, retained traces and separate evaluation of argument accuracy.
