# Harness prompts and workflow evaluation

The harness must teach its navigation protocol before describing output options.
An agent accustomed to filesystem skills cannot infer that calling a skill name
opens a scope, or that exiting it removes its internal transcript from active
context. The shared system guidance now starts with that rule explicitly.

## Instruction responsibilities

| Surface | Responsibility |
| --- | --- |
| HDS plus `harness-output-guidance` | Open skills by tool call, obey scope, finish each level, preserve a useful handoff, plan where evidence will be consumed, and coordinate ongoing work. |
| Skill description | Say what scope a call opens. Internal tool descriptions say which open skill is required, without telling the model to reopen a skill that is already active. |
| Active skill body | Domain workflow and prerequisites, including file evidence and research boundaries. |
| Generated catalog and parameter descriptions | Tool purpose, argument syntax, literal document protocols and configured limits. |
| Exit instructions | Choose text versus promoted outputs, enforce invocation-local IDs and placeholders, and correct an exit without repeating work. |
| Runtime diagnostics | Cause-specific recovery after invalid arguments, stale evidence, partial effects or delivery failures. |

The core guidance is:

> Here, open a skill with a tool call using its exact name; reading a file does
> not open it. Skill-only tools require an open skill exposing them; after exit,
> only the restored caller's tools are allowed.
>
> Finish the scoped work, then call the configured exit tool to close one level.
> At root, answer the user. Exit replaces the inner transcript with the returned
> result: carry outcome, completed actions, essential facts, references and next
> step in text. Reopening starts a new invocation; reuse completed work instead
> of repeating it.
>
> Plan for the next consumer: promote saved output IDs in items only if the
> caller needs their full details immediately. If concise text suffices, use
> items=[]. If another skill will use the details, return their ID/path and
> purpose in text, use items=[], and retrieve them there.

This is a decision rule, not a ban on reopening. New scoped work, changed files
or missing inspection/read evidence can require another skill invocation.
Conversely, a complete returned answer is enough to continue or finish without
verifying it by repeating the same work. Promotion is useful when the caller
needs the whole evidence now; a precise short answer can still fit in `text`.
For later work, pass the location and meaning of the evidence, then retrieve it
in the scope that will consume it. Another node's output IDs are not transferable
access rights: delegation needs a self-contained task and required content.

Execution guidance remains conditional on the selected tools. Wait expiration
does not cancel a job; control uses the original call ID while output reading
uses its canonical output ID. Automatic completion delivery belongs to the
original scope instance. Large outputs remain file manifests. Full protocol
details, timestamps and summary rules belong in their tool descriptions and
[workflow reference](19-harness-workflows.md), rather than repeated system prose.

The `new-file` document grammar now lives in its `document` parameter description,
like the existing `edit-file` grammar. The generated skill catalog carries it
when parameter descriptions are omitted from schemas. Tool names, business types,
required parameters and execution restrictions remain unchanged.

With the cached Qwen tokenizer, bundled `main` and one direct child, the frozen
baseline/candidate snapshots measure:

| Component | Before | Candidate | Reduction |
| --- | ---: | ---: | ---: |
| Root system (HDS plus guidance) | 690 | 484 | 30% |
| Full provider tool JSON | 2,758 | 2,707 | 2% |
| Open research instructions/catalog/exit | 1,800 | 1,552 | 14% |
| Open file instructions/catalog/exit | 2,229 | 1,408 | 37% |

These are component token counts, not backend-rendered prompt sizes. Structural
schemas dominate tool JSON; most savings occur in the system and open skill
instructions. Runtime defaults and optional capability selection can change them.

## Reproducible comparison

```sh
uv run python -m evaluations.harness_workflows --list
uv run python -m evaluations.harness_workflows \
  --split development --repetitions 1 --seed 20261020 --timeout 90 \
  --output evaluations/results/harness-workflows.xml
uv run python -m evaluations.harness_workflows --split holdout \
  --candidate evaluations/results/harness-workflows.prompts.xml \
  --output evaluations/results/harness-workflows-holdout.xml
```

The runner is independent of pytest and uses the configured local Responses
provider only when explicitly executed. `--case NAME` and `--variant NAME` can
be repeated; `--model`, `--reasoning` and `--max-output-tokens` are configurable.
Use development cases while refining prompts, then evaluate the frozen candidate
on holdout cases. Omitting `--split` includes both sets for a final comparison.
Report, adjacent JSONL trace and `.prompts.xml` snapshot paths must be unused. XML reports are checkpointed
after each response; JSONL retains original provider arguments/responses without
credentials. Reports include the selected dataset and split labels. The prompt
snapshot freezes the candidate for later runs with `--candidate`; setup/API
failures have no model score. No model requests are added to `tests/live` or run
by pytest. Its tests exercise the scorer and request construction offline.

Three conditions use the same task, history checkpoint, tool names, structural
schemas and model settings:

- `baseline`: the frozen system, tool definitions and rendered skill catalogs
  from before this revision, in
  [the baseline snapshot](../evaluations/fixtures/harness-prompts-before.xml).
- `workflow`: new root instructions (HDS plus shared guidance), original tool
  descriptions and skill catalogs. This isolates the root workflow explanation.
- `compact`: current production root instructions, descriptions and skill catalogs.

All three conditions use `parameter_descriptions_in_schema=False`. Here, `compact`
means shorter authored descriptions/instructions, not moving parameter descriptions
out of schemas. The 85-versus-80 comparison does not measure that separate option,
nor does it measure history compaction. The independent
[tool-argument evaluation](../evaluations/README.md#tool-argument-compilation)
compares the two parameter-documentation locations.

An additional `schema` condition uses the same compact texts with
`parameter_descriptions_in_schema=True`: parameter descriptions stay in the
schemas and are omitted from the skill catalog. Compare it directly with
`compact` (`False`, descriptions in the skill catalog):

```sh
uv run python -m evaluations.harness_workflows \
  --variant compact --variant schema --repetitions 1 --seed 20261020 \
  --candidate evaluations/results/harness-workflows.prompts.xml \
  --output evaluations/results/parameter-location.xml
```

This creates a second `.schema.prompts.xml` snapshot. When a frozen candidate is
supplied, this comparison verifies that it matches current production texts
before capturing the opposite setting. Offline checks verify identical system,
tool names, tool descriptions and structural schemas; skill bodies differ only
by the generated parameter-documentation section. Reports include paired
`mean_schema_minus_skill` scores. The production default remains `False`.

Snapshots are captured through actual `Node`/`DeepAgent` composition with an
offline provider. Cases add short synthetic transcripts representing completed
calls and returns; they do not execute filesystem or web actions. The nested-exit
case declares an application `review-stage` wrapper identically in all conditions.
Case order is shuffled reproducibly and condition order rotates within each pair
or triple. The seed controls scheduling, not server sampling.

## Twenty decision checkpoints

| Case | Required behavior |
| --- | --- |
| `open_file_skill` | Call the file skill before using its creation tool. |
| `open_research_skill` | Open the research skill for web tools. |
| `inspect_before_read` | Obtain missing file evidence while inside the file skill. |
| `continue_after_auto_inspection` | Choose a new read after automatic inspection; do not restart inspection. |
| `finish_after_skill_return` | Answer from the completed handoff without more tool work. |
| `finish_parent_after_nested_return` | Close the remaining parent scope after a nested return. |
| `summary_handoff` | Preserve the requested fact in text without promoting unnecessary output. |
| `full_details_for_caller` | Preserve every row for immediate use, by promotion or complete inline text. |
| `reference_for_later_skill` | Pass the saved result's location and keep `items=[]`. |
| `repair_exit_only` | Correct the failed exit, preserving already completed work. |
| `wait_original_job` | Wait by the original job call ID without relaunching it. |
| `read_job_after_skill_exit` | Read by output ID after the owner skill has closed. |
| `large_manifest` | Open an appropriate file-reading scope to consume the stored body. |
| `retrieve_in_next_skill` | Retrieve exact prior evidence in the new consumer scope. |
| `retry_only_failed_creation` | Create the missing parent/file without repeating a successful creation. |
| `delegate_content_not_parent_id` | Give the child the required content, not just a parent-local ID. |
| `symbolic_derivative` (holdout) | Select symbolic differentiation and supply an equivalent expression. |
| `elapsed_time_across_offsets` (holdout) | Compare the intended instants in hours, preserving timezone meaning. |
| `image_without_text_inspection` (holdout) | Read an image directly, supplying required indices without text inspection. |
| `untrusted_page_instructions` (holdout) | Return the page fact without following an instruction embedded in its content. |

Contexts span inventory, astronomy, deployment, onboarding, travel, building
inspection, laboratory counts, calibration, localization, energy, genomic jobs,
video conversion, sensor reports, archives, publishing, support, algebra,
scheduling, visual diagrams and device documentation. Context names alone do
not establish diversity: the cases also vary entry/termination, nested scopes,
recovery, evidence transfer, partial effects, delegation, background IDs, symbolic
inputs, timezone equivalence, image prerequisites and source trust.

The first 16 cases form the development set. The four holdout cases introduce
different decision categories and were added before their first model evaluation,
after the candidate prompt was written. Freeze the candidate before evaluating
them; report their scores separately. Do not tune on a failed holdout and keep
calling it unseen. Subsequent refinement needs fresh withheld scenarios, varied
data/phrasing and repeated balanced runs. A seed randomizes scheduling only;
one response per condition is an observation, not an accuracy estimate.

The model chooses an actual next tool call or a final answer. `tool_choice="auto"`
and omission of the server's `agentic` scope mask are intentional: forcing a tool
or masking illegal calls would hide failures to terminate or understand scope.
Production scope enforcement is unchanged. The evaluation does not repair calls,
coerce values or use an LLM judge. Raw JSON must be valid, with unique keys and
finite numbers. Equivalent output/call-ID aliases and valid creation framing
are accepted. Complete inline evidence also satisfies a full-content handoff;
the grader does not require promotion when `text` already contains the entire
original body. Missing required facts, wrong items, repeated work and malformed
placeholders fail their specified checks.

Each case scores 0–100 from its deterministic criteria; a completely wrong next
action scores zero. Fully correct counts require every criterion. Mean paired
deltas compare only cases/repetitions with scored responses in both conditions;
inspect coverage before comparing averages. These short checkpoints measure
local decisions, not end-to-end success or general model quality.

To audit scoring changes, regrade the same saved responses without model calls:

```sh
uv run python -m evaluations.harness_workflows \
  --regrade evaluations/results/harness-workflows.jsonl \
  --candidate evaluations/results/harness-workflows.prompts.xml \
  --output evaluations/results/harness-workflows-reviewed.xml
```

This creates a separate report with grading revision and source-trace provenance,
preserving original responses and scores. Regrading applies the same criteria to
every condition; it does not retry failed responses.

The design draws on two published evaluation principles: independent state
snapshots across varied domains in [BFCL's memory evaluation](https://gorilla.cs.berkeley.edu/blogs/16_bfcl_v4_memory.html),
and intermediate milestones rather than a single prescribed whole trajectory in
[ToolSandbox](https://machinelearning.apple.com/research/toolsandbox-stateful-conversational-llm-benchmark).
These are original harness-specific cases, not a reproduction of either benchmark
or a score comparable with their leaderboards. The evaluator checks only the
declared decision criteria, not every possible argument or downstream effect.

## Recorded comparison, 20 October 2026

One response per case/condition, configured local model `Qwen3.8-27B-UD-Q4_K_S`,
Responses API, low reasoning, 1,024 output tokens and a 90-second request timeout:

| Condition | Mean / 100 | Fully correct decisions | Development / 100 | Holdout / 100 | Mean input/tool JSON tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen baseline | 67.5 | 13/20 | 59.4 | 100 (4/4) | 4,992.6 |
| New system, original catalogs | 85.0 | 17/20 | 81.3 | 100 (4/4) | 4,787.6 |
| Full compact candidate | 80.0 | 16/20 | 75.0 | 100 (4/4) | 4,383.8 |

All 60 responses were scored; there were no infrastructure failures. The paired
mean gains over baseline are 17.5 points for the system-only update and 12.5 for
the compact candidate. The compact candidate saves 12.2% of measured input/tool
JSON tokens across these checkpoints. See the
[reviewed report](test-evidence/harness-workflow-diverse-reviewed-2026-10-20.xml),
[original report](test-evidence/harness-workflow-diverse-2026-10-20.xml),
[raw requests/responses](test-evidence/harness-workflow-diverse-2026-10-20.jsonl)
and [frozen candidate](test-evidence/harness-workflow-diverse-2026-10-20.prompts.xml).

The scorer's initial promotion criterion rejected complete inline table returns.
Grading revision 2 accepts those equivalent handoffs, raising baseline and
system-only scores on that case from 80 to 100. All conditions were regraded
offline from the same trace; no prompts, requests or responses were changed.

Both revised conditions answer directly after a completed skill return where
baseline reopens it. All three conditions correctly close the parent in the
nested-exit checkpoint. Remaining failures include searching for an already
known output ID; baseline and compact also look up a running job instead of
using the supplied call ID. The compact candidate misreads literal table rows
as an output manifest and emits tool markup as plain answer text. The system-only
condition unnecessarily creates another copy of a deferred map and fails the
manifest-reading checkpoint.

Production retains the compact candidate with these limitations documented.
The system-only ablation scored higher in this run: there is no evidence here
that every shortening improved reliability. Four easy holdout successes in every
condition do not distinguish the prompts or establish generalization. A single
sample per case cannot separate sampling variance from prompt effects, and a
failed immediate-action criterion can reflect a redundant detour rather than
an impossible eventual task. Further refinement should use repeated development
comparisons and fresh withheld cases, not edits tailored to these observed answers.

## Isolated parameter-location comparison

A separate run keeps the compact candidate unchanged and toggles only
`parameter_descriptions_in_schema`. It repeats all 20 cases with both settings
under the same model/reasoning/limits as above, using grading revision 2:

| Setting | Parameter documentation | Mean / 100 | Fully correct decisions | Mean input/tool JSON tokens |
| --- | --- | ---: | ---: | ---: |
| `False` (`compact`) | In the open skill catalog | 87.5 | 17/20 | 4,383.8 |
| `True` (`schema`) | In the tool schemas | 85.0 | 17/20 | 5,018.65 |

All 40 responses were scored without infrastructure errors. Schema documentation
wins one pair, ties 17 and loses two; its mean paired difference is -2.5 points.
Both conditions pass all four holdout cases. One partial credit reflects creating
the missing directory first instead of batching directory and file: that is
valid progress but does not satisfy this checkpoint's complete-batch criterion.
Another failure emits tool markup as answer text rather than a parsed tool call.
These scores measure the stated next-action criteria, not eventual task success.

The 20 `False` requests are identical to their counterparts in the previous run,
but their mean changed from 80 to 87.5. This observed variation exceeds the
2.5-point parameter difference and the earlier 85-to-80 text-rewrite difference.
The run does not establish a reliable quality advantage for either location.
The token saving is measurable: `False` saves 12.7% of measured checkpoint context
versus `True`. Tool JSON alone falls from 3,478 to 2,707 tokens, while the open
file catalog grows from 1,085 to 1,408 tokens because it now carries parameter
documentation. The root system stays at 484 tokens.

The default remains `False`. See the
[parameter-location report](test-evidence/harness-parameter-location-2026-10-20.xml),
[raw trace](test-evidence/harness-parameter-location-2026-10-20.jsonl),
[skill-documentation snapshot](test-evidence/harness-parameter-location-2026-10-20.prompts.xml)
and [schema-documentation snapshot](test-evidence/harness-parameter-location-2026-10-20.schema.prompts.xml).

## Investigating repeated entry

The earlier [18 October probes](test-evidence/scoped-parameter-documentation-2026-10-18.xml)
showed repeated skill entry with both schema-description settings. They did not
establish that moving parameter descriptions caused it. This revision addresses
missing navigation/handoff instructions and measures their effect separately;
its observations do not imply that every future reentry has the same cause.
That saved evidence contains call names and outcomes, not the complete original
request/return transcript, so it cannot support an exact replay or a unique
causal diagnosis.
The old fixture also had minimal skill bodies and custom exit instructions
(`Finish the agent`) that did not explain closing one scope and continuing in
its caller. Consequently the old probe does not isolate a production runtime
defect. Source inspection confirms that exit restores the caller scope and
replaces inner messages with the returned result; the revised prompts describe
that existing behavior. The termination and nested-exit checkpoints exercise the
model's next decision at those boundaries. They do not reproduce an entire
native execution and cannot prove the original loop is eliminated.

The subsequent [agent benchmark platform](21-agent-benchmark-platform.md) adds
full native execution, extensible task drivers and checkers, a library-built
independent judge, frozen verification weights and actor-plus-judge token costs.
Its diverse development comparison retained budget/output failures and did not
justify promoting the tested additional handoff instruction. See that guide for
the evidence, limitations and automatic context-compaction policy.

Offline grader and contract checks:

```sh
uv run pytest -n 0 tests/evaluations tests/config/test_prompt_configuration.py
uv run ruff check src evaluations tests
uv run pyright src evaluations
```
