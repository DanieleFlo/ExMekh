# Agent benchmark platform

`evaluations.benchmark` compares complete agent attempts, their delivered artifacts,
and their inference costs. It combines deterministic checks with a separate
`DeepAgent` evaluator built from this library. Model campaigns are standalone
commands. Pytest only tests the platform with scripted providers and bounded local
programs; it never starts these campaigns.

## Run a campaign

From the repository root, with the configured local engine running:

```sh
uv run python -m evaluations.benchmark --list
uv run python -m evaluations.benchmark --case scoped_sum --variant current \
  --output evaluations/results/smoke.xml
uv run python -m evaluations.benchmark --split development --repetitions 3 \
  --judge-reasoning low --output evaluations/results/development.xml
# Freeze prompts, criteria and weights before opening the holdout results.
uv run python -m evaluations.benchmark --split holdout --repetitions 3 \
  --judge-reasoning low --output evaluations/results/holdout.xml
```

The runner reads `.env`, uses `LocalProvider` with the Responses API, disables SDK
retries, and accepts `--model`, `--judge-model`, `--reasoning`, `--judge-reasoning`,
`--timeout`, `--seed`, repeated `--case` and repeated `--variant`. Actor and judge
default to `low`, the minimum enabled reasoning level. A distinct judge model can
reduce shared-model bias. The default schema-description policy remains `False`.

Every trial gets a fresh workspace and node context. Variants run consecutively
within each task/repetition, with their order rotated; task order is shuffled.
The seed controls scheduling, not model sampling. Existing report paths are
rejected. Reports are checkpointed after each trial, and inference records are
flushed after each completed or failed provider call.

The native scaffold uses a short application HDS, a selected subset of bundled
capabilities, and an added bounded Python runner in `file-processing`. It retains
the library's production harness guidance. This is an application-level benchmark,
not a reproduction of every default application configuration.

The default comparison uses `suites/handoff_variants.json`: unchanged guidance
versus a short statement that skills are scopes of the same agent and completed
verification should be reused. `suites/prompt_variants.json` retains the earlier
ID/defaults/validation addendum for separate experiments; it is not combined with
the handoff condition.

## Tasks and outcome checks

The bundled suite contains twelve new tasks across different domains:

| Task | Split | Evidence |
| --- | --- | --- |
| Scoped arithmetic | Development | Exact answer and explicitly requested skill/call/exit order |
| Time-zone interval | Development | Numeric result and use of the requested date tool |
| Inventory reconciliation | Development | Signed quantities, decimal prices and JSON totals |
| Weighted routing | Development | Executed functions on multiple graphs; explanation quality |
| Two-dimensional fluid box | Development | Executability, numerical properties and physical reasoning |
| Constrained delivery choice | Development | Feasibility and explanation of alternatives |
| Untrusted source instructions | Development | Correct fact and unchanged workspace |
| Thermal sensation | Development | Length constraint and scientific explanation |
| Queue simulation | Holdout | Executed FCFS examples and stated assumptions |
| Signal smoothing | Holdout | Executed causal filter and explanation of its tradeoff |
| Incident diagnosis | Holdout | Evidence, uncertainty and useful next actions |
| Compaction archive | Holdout | Required facts and an actual applied compaction event |

Coding tasks do not prescribe a history or a particular algorithm. Checks observe
the declared files and execute their APIs. Only tasks that explicitly request an
exact sequence receive a sequence check. The fluid probes verify finite grids,
nonnegative density, closed normal wall velocities, bounded mass change and
nontrivial evolution. These are smoke checks, not proof of a correct fluid solver;
the physical model receives a separate semantic assessment.

The `checkpoint` driver also adapts cases from the earlier
[workflow benchmark](20-harness-prompts-and-evaluation.md). A checkpoint task uses
`parameters: {"case": "inspect_before_read"}` and the `workflow` checker. Its
score measures the next decision, whereas the `native` driver executes a real
`DeepAgent` trajectory. Do not treat their averages as the same measurement.
This adapter observes library-resolved arguments. Use the earlier standalone
workflow/tool-argument runners when raw argument JSON before provider coercion is
the measurement of interest.

## Fixed grading policy

Each criterion declares an importance, a verification class, and a reason for that
class **before** candidate execution. The evaluator cannot choose these weights:

| Verification | Algorithm weight | Judge weight | Intended use |
| --- | ---: | ---: | --- |
| `exact` | 1.00 | 0.00 | A fixed check decides the entire criterion |
| `strong` | 0.80 | 0.20 | Mechanical evidence is strong but does not settle adequacy |
| `partial` | 0.35 | 0.65 | Mechanical checks only cover a limited part |
| `semantic` | 0.00 | 1.00 | Meaning, explanation or design requires content assessment |

For criterion i, `score_i = algorithm_weight_i * algorithm_score_i +
(1 - algorithm_weight_i) * judge_score_i`. Overall quality is the mean weighted by
criterion importance. These initial weights express a reviewable policy, not
empirically calibrated confidence probabilities. Prefer separating executability
and conceptual correctness into different criteria over giving one vague
criterion an arbitrary blend.

Success is separate from quality: **every required criterion must reach its own
threshold**. A high explanation score cannot rescue a failed required executable
check. Without required criteria, success remains unspecified. Missing analytical
or judge evidence leaves quality unknown; remaining weights are never increased.
Missing supplementary evidence does not hide a known success or failure on the
required criteria. Reports count these successes independently of quality coverage.
Exact criteria cannot be scored or overridden by the judge.

The native task also has a completion gate: an actor that does not finish within
the declared budgets receives zero and fails that gate. This does not imply that
every intermediate file is incorrect; artifacts remain available for diagnosis.
Grader or infrastructure failures instead leave quality unknown. Keep completion,
individual checks and artifact usefulness distinct when interpreting a score.

Supplementary criteria can lower quality without failing a required gate. Task
authors must mark essential requirements as required; a success count is not a
claim that every supplementary criterion received full marks.

The evaluator receives the task, criterion descriptions, answer, artifacts,
call evidence and analytical observations. It does not receive the variant name,
candidate prompt or algorithm's numeric marks. Its final JSON verdict requires criterion IDs,
integer scores from 0 to 100 (the prompt requests increments of five), valid evidence IDs and
a short rationale. A fixed validator rejects duplicate, missing or
invented IDs, extra verdict fields and coerced/noninteger scores. The verdict file
can only be created once. Task text,
source and results are explicitly untrusted evidence. This reduces simple
contamination but does not prove immunity to judge prompt injection or bias.

## Token cost and compaction

`MeteredProvider` records each call outside the agent's mutable history, including
structured calls, child calls, output summaries and history compaction calls made
through that provider. Skill exits and compaction cannot erase past costs.
The variant addendum applies to actor inference, not internal compaction prompts.
Provider requests are copied without re-expanding multimodal attachments or
losing interactive handlers and structured-response contracts.
Consumers use `get_usage_totals()`, `get_tool_calls()` and detached diagnostic
snapshots from `get_inferences()`; they cannot edit the provider's ledger. The
elapsed inference budget begins on its first call, independently for actor/judge.

Actual provider usage supplies input, output and total tokens. Terminal streaming
usage is retained even if response parsing rejects an incomplete completion.
Actor and evaluator costs appear separately and together; failed attempts still
count. Missing provider usage is **unknown**, not zero. Reports show complete-usage
coverage and known token totals; aggregate totals and tokens-per-success remain
unknown when any trial lacks usage. Local tokenizer estimates are separate and
are never silently substituted for measured costs.
Actor and judge means each use their own complete-usage coverage, so unavailable
judge usage does not erase measured actor cost, or vice versa.

Reports include quality, success, usage coverage, paired quality/token deltas,
tokens per success and a configurable cost-adjusted quality:
`quality - token_penalty * total_tokens / 1000`. The default penalty is one point
per thousand actor-plus-judge tokens. It is an explicit utility choice, not a
universal ranking. Compare actor cost separately when production efficiency is
the question, since judge cost can dominate a short task. Tokens are gross usage;
the platform does not infer monetary prices or discounts for cached tokens.

Native actor nodes, including children, enable automatic history compaction by
default before inference at `compaction_threshold=0.9` of `max_context_tokens`.
The same prepared-request estimator counts variant instructions and schemas in
both threshold decisions and inference checks. The library's `RequestBudget`
also reserves the task's full `max_output_tokens` and can force compaction earlier
than 90% to keep that space available. The provider applies the same prompt limit
even when compaction is forbidden. Its bounded compactor splits internal requests when needed. Protected
input that still cannot fit produces an explicit budget failure; it is never
silently dropped. This policy belongs to the benchmark driver and does not change
the library's application-owned policy defaults.

Tasks can explicitly use `compaction="forbid"` for an isolated comparison.
The archive task additionally schedules compaction after a call boundary and
requires evidence that it was applied. All compactor and summarizer calls consume
the same actor ledger and call/token limits. The report counts compaction calls.
The single-pass judge rejects oversized evidence rather than summarizing protected
rubric/input text. Compaction cannot cure a completion that exhausts its separate
output-token limit. Budgets limit calls, context estimates,
total tokens, response output and elapsed time checked at inference boundaries;
the HTTP read timeout handles stalled streams. Context estimates are not exact
server KV-cache measurements.

Calls are logical library provider calls. Interactive pause/resume HTTP exchanges
belong to one logical call and use its terminal provider usage; the platform does
not count a continuation twice. An interrupted campaign remains `complete=false`;
its partial total must not be compared with a completed campaign's total.

## Extend the platform

`--suite path.json` accepts any nonempty version-1 suite; no logic depends on the
bundled task count. Each task declares `id`, `domain`, `prompt`, `criteria`, optional
seed `files`, declared `artifacts`, `parameters`, `split`, `driver` and `limits`.
The Pydantic definitions reject duplicate IDs and inconsistent verification modes.
`--profiles path.json` accepts any distinct variants with a system addendum and
schema-description policy. Both configurations are embedded in the report.

For another task type, a trusted Python plugin exports:

```python
from evaluations.benchmark.models import Check, Observation

def register(components):
    def run(task, variant, directory, provider):
        # Execute the task with the supplied metered provider.
        return Observation("completed", answer="observed result")

    def verify(task, observation, workspace, parameters):
        return Check(100 if observation.answer == parameters["expected"] else 0,
                     {"actual": observation.answer})

    components.drivers.register("my_driver", run)
    components.checkers.register("my_check", verify)
```

Run with `--plugin my_package.my_benchmark`. Plugins are trusted application code;
they must route model calls through the supplied provider for complete accounting.
They can replace `components.judge` while preserving the independent rubric.

The native driver's generated Python tool uses Linux user/network/PID namespaces,
Landlock filesystem restrictions, seccomp and resource limits. Candidate source
is readable but cannot be overwritten by its execution; network sockets and
process execution are denied. Only standard-library execution is supported.
The worker requires Linux with these kernel facilities and `libseccomp`; if setup
fails, execution is unavailable and no unrestricted fallback runs. Timeouts,
nonzero exits and truncated output remain explicit. This bounded process is not
a VM and does not promise protection against kernel vulnerabilities.
Readiness uses a private descriptor closed before candidate code starts; text
printed by a candidate cannot impersonate a launcher failure. Interruption kills
and reaps the process group before the exception propagates.

## Evidence and refinement

The XML report freezes tasks, criteria, profiles, model/reasoning settings and
source hashes. Adjacent JSONL records preserve inference requests, responses,
errors, timings and usage; per-trial directories retain declared artifacts and
verdicts. XML uses the shared renderer. JSONL is the intentional provider-history
exception. Reports and generated workspaces under `evaluations/results/` are
ignored; copy selected evidence into `docs/test-evidence/` when publishing results.

Campaign orchestration lives in `campaign.py`, contracts in `models.py`, and
statistics/XML persistence in `reporting.py`. `probes.py` owns independent program
execution outcomes; checkers score them without interpreting worker internals.
All input probes run after individual program failures; an infrastructure failure
instead leaves the check unavailable. Failed probes retain their original input
position and contribute zero without shifting subsequent expected results.

Refine generic instructions using development failures, freeze the candidate,
then evaluate untouched holdout domains. Inspect algorithm/judge disagreements and
artifacts before editing prompts. Repeat comparisons to assess variability; one
run cannot establish superiority. A shared actor/judge model, a small holdout and
synthetic tasks limit generalization. Human calibration remains necessary for
semantic scores. Do not change rubrics to favor the candidate after seeing results.

The design combines outcome and trajectory evidence as discussed in
[Anthropic's agent evaluation guide](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents),
uses flexible milestone reasoning rather than one prescribed trajectory as in
[ToolSandbox](https://machinelearning.apple.com/research/toolsandbox-stateful-conversational-llm-benchmark),
and accounts for limitations of model judges described in
[MT-Bench's judge study](https://arxiv.org/abs/2306.05685). The tasks here are original
local fixtures; their scores are not scores on those external benchmarks.

## Pilot audits

The first inventory pilot accepted a natural-language output specification but
checked an unstated JSON key name. An otherwise correct `units_by_sku` answer was
therefore scored zero against `units`. That pilot was stopped and excluded from
prompt selection. The final task explicitly names both keys; the compared
profiles were not changed to favor a particular representation.

An early judge used a nested `submit-verdict` tool. On this local engine, numeric
generation sometimes repeated decimal zeros or timed out after reaching the
correct conclusion in reasoning. A discrete score schema succeeded on a small
diagnostic but remained unreliable on realistic evidence. The final evaluator is
still a library `DeepAgent`, but returns one raw JSON verdict. The platform validates
and records it directly, without a second inference to acknowledge a tool result.
Failed pilots retain their missing-usage flags; they do not become zero-cost runs
or substitute scores. Their exploratory costs are separate from the frozen
comparison's costs.

A later offline integration test found that isolated Python could not import a
sibling workspace module. The runner now adds the bounded workspace to its import
path after applying kernel restrictions; external access remains restricted.
Affected coding attempts were stopped and rerun. The actor's execution tool was
also reduced to a single script-path argument. Results from different scaffold
versions must retain their individual source-report provenance.

## Exploratory development comparison

The later code review separated provider accounting, execution probes and report
generation. It corrected missing-null JSON keys, independent probe scoring,
optional-evidence success gates and zero-token accounting. Output room now uses
the declared completion limit, and unfinished responses use a typed error.
Previously frozen campaign reports were not rewritten or silently regraded;
their source versions and limitations remain part of the evidence below.

Eight development tasks compared the unchanged `current` profile with the frozen
generic `handoff` addendum, one repetition each. Both actor and judge used the local
`Qwen3.8-27B-UD-Q4_K_S` model with reasoning `low`.

| Profile | Mean quality / 100 | Successful tasks | Actor tokens | Judge tokens | Total tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| current | 57.19 | 4 / 8 | 157,456 | 4,593 | 162,049 |
| handoff | 49.06 | 4 / 8 | 209,208 | 3,874 | 213,082 |

The candidate consumed 31.5% more total tokens and scored 8.13 points lower.
It is retained as an experiment, not adopted into production instructions.
These are descriptive observations with one sample per pair, no confidence claim
and no holdout-based selection. Four noncoding tasks used the first corrected
scaffold; four remaining tasks used the sibling-import fix and simplified execution
tool. Every task's two profiles used the same scaffold version. This is not a
single-scaffold estimate of general agent performance.

Budget and incomplete-output failures remain in both score and cost totals.
For example, the current route task retained code that passed all three independent
API probes, but never completed within its call budget: its official score remains
zero. Both fluid attempts exhausted 4,096 output tokens on the first response,
with only about 2,200 input tokens. Those failures do not demonstrate context
exhaustion. All selected calls reported usage. Exploratory interrupted pilots
remain separate, including their unknown-usage flags.

The [comparison review](test-evidence/agent-platform-comparison-2026-10-20.xml)
records selection, hashes, paired statistics and the offline artifact diagnostic.
Its sources are the [partial development campaign](test-evidence/agent-platform-development-v3-2026-10-20.xml)
and [remaining development campaign](test-evidence/agent-platform-development-final-2026-10-20.xml),
with adjacent JSONL inference traces. These historical campaigns explicitly
forbade compaction except for the archive fixture and observed no compaction calls.
The 90% default was added afterwards; their frozen manifests must not be interpreted
as measurements of that new policy.

Offline platform tests cover the threshold just below/at/above 90%, unchanged
request objects, child-node configuration, actual library compactor calls with
scripted providers, preserved usage and exhaustion of the shared call budget.
These validate integration and accounting, not a new measured model-quality gain
from automatic compaction.
