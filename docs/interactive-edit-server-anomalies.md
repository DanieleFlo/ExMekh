# Interactive edit: live server observations

Observed on 2026-10-09 and rerun on 2026-10-11 with `Qwen3.8-27B-UD-Q4_K_S`, DS4 Responses interactive v1,
the configured running server and OpenAI SDK 3.3.0. These are open observations,
not established root causes. In particular, a model-looking symptom can originate
from logits masking, native parameter framing or the selection/replacement sampling
frontier. Prompt changes alone do not qualify a server fix.

No server code, masking, KV state or sampling implementation was changed by this
client task. The client preserves the generated strings and does not trim markers,
complete a truncated marker, convert ordinary replacement content to deletion, or
silently fall back to a normal tool call.

## S1: deletion request produces a rewrite

The complete node workflow inspected and read the two physical lines of
`notes.txt`, whose exact bytes were `b"  old value\r\nKEEP\n"`. The user required
`old_text="@ex_meckh_all", new_text="@ex_meckh_delete"`. The snapshot exchange was
accepted, but the completed call instead contained `new_text="KEEP"`. The client
performed the requested argument-level rewrite, leaving a four-byte file, so the
live deletion test failed. Repeated recovery could exhaust its 14-inference budget.

This completed call is a valid ordinary rewrite. It demonstrates an unfulfilled
deletion request; it does not by itself prove a violation of the edit constraints.
Compare the model's unconstrained next-token choice with the applied replacement
mask and restored decode state before attributing it to model behavior.

## S2: newline and truncated delete marker are completed as ordinary text

An isolated call, without skill routing or filesystem dispatch, required the same
marker pair and supplied the exact snapshot
`"  value\nprefix value\naaa\nvalue"`, UTF-8, four physical lines. The server
completed `old_text="@ex_meckh_all"` and returned replacement strings
`"\n@ex_meckh"`, `"\n@ex_meckh\n"` or `"\n@ex_meckh\ndelete"`. Explicit inline-marker instructions,
temperature zero, ordinary/default reasoning and per-argument descriptions did
not establish a successful deletion. The full node reproduced the latter value.

Captured provider input, pause/reply and output are retained in
[the isolated trace](test-evidence/delete-marker-isolated.json) and
[the full node trace](test-evidence/delete-marker-full-node.json). These are
structured SDK/provider observations, not byte-for-byte HTTP recordings.
The permanent isolated test also retains [actual ordinary request bodies and
parsed raw events](test-evidence/delete-marker-wire-ordinary.json): the final
`response.completed` contains `"\n@ex_meckh\ndelete"` after 120 output tokens.
This run used reasoning `none`, temperature zero and a 256-token output budget.

Potential server checks:

- Verify the replacement-opening frontier and forced/native framing tokens;
  a token spanning the header/body boundary must not skip the dynamic policy.
- Verify that the previous selection is decoded exactly as all, and that the
  authorized existing-file state enables delete from its first token.
- Check the completion rule for the bare reserved stem `@ex_meckh`. The observed
  string lacks the underscore of the full family prefix; accepting it as ordinary
  data may expose an incomplete-marker gap. Preserve valid ordinary newlines;
  banning all replacement-leading newlines would change the contract.
- Compare target-only and MTP paths and ensure replacement uses the restored
  sampling state with an updated mask after every token.

The client had the same incomplete-stem gap and has been corrected: these exact
replacement strings now return `invalid_file_edit` without a write. Deterministic
selector and actual-SDK/node regressions cover the observed payload. This client
guard does not resolve the server's generation problem.

## S3: marker-only schema ends with incomplete_generation

An isolated schema kept all three roles as required strings, adding single-value
enums for old/all and new/delete. After a valid path pause and provide reply, the
server returned `interactive.failed`, code `incomplete_generation`, message
`interactive generation ended before completion`. One recorded pause used 44 of
the 256 output tokens and 694 input tokens. In the permanent reproduction,
[the captured wire events](test-evidence/delete-marker-wire-enum.json) show a
44-token pause followed by failure at the full 256-token budget (757 input
tokens). Both isolated permanent tests failed in a combined 56.99 seconds.
No executable call was returned for the enum case.

This is a separate diagnostic: inspect the intersection between the native/schema
grammar and dynamic edit policy. A budget failure alone does not prove that their
intersection is empty. The enum is confined to the diagnostic test; the production
tool has ordinary string fields, without an operation enum or extra arguments.

## C1: SDK response metadata compatibility

Some terminal responses lacked `parallel_tool_calls`, `tool_choice`, `tools` or
the SDK-required `input_tokens_details.cache_write_tokens`. Strict SDK validation
initially failed. The adapter now supplies only unused compatibility defaults and
then performs normal SDK validation; raw argument types, duplicate keys, IDs and
frozen authorization remain independently checked. A missing cache-write counter
is not evidence that the server never writes a prompt cache. Aggregate input,
output and total usage come from the server's final cumulative values.

## S4: deletion marker emitted without its initial character (2026-10-11)

The clean-code review reran the live workflow against the running server. Literal
editing and creation passed, as did the bounded old-text attack. Deletion did not
complete within the 14-turn workflow budget. Five completed edit calls used
`old_text="@ex_meckh_all"` and `new_text="ex_meckh_delete"`: the latter is missing
the initial `@` of the configured deletion marker. The file remained present,
containing exactly the ordinary literal `ex_meckh_delete`.

The [first response and snapshot exchanges](test-evidence/clean-code-delete-workflow-response-2026-10-11.json)
and [the following request with its actual tool result](test-evidence/clean-code-delete-workflow-followup-2026-10-11.json)
preserve the prompt, arguments and successful `operation=edit` result. These
strings describe this recorded run, rather than runtime defaults.

This is a failure to fulfill the requested deletion. The returned replacement is
valid ordinary text under the contract, so this observation alone does not prove
a masking bypass or identify whether the generation frontier or the model caused
the missing character. Inspect native parameter framing and the replacement
mask at the first marker token. The client must not infer an operation from a
near-marker string, restore characters or rewrite generated arguments: doing so
would prevent valid literal content and conceal the generation failure.

The isolated checks reproduced the earlier defects in this run: the
[ordinary schema](test-evidence/clean-code-delete-marker-wire-2026-10-11.json)
returned `new_text="\n@ex_meckh\ndelete"`, while the
[single-value enum schema](test-evidence/clean-code-delete-marker-wire-enum-2026-10-11.json)
ended with `interactive.failed` / `incomplete_generation`. The client did not
dispatch either diagnostic call. The 30-second attack lasted 30.56 seconds,
attempted two generations and independently validated one complete constrained
call, without writing its attack file.

An earlier broad run used a 120-second request timeout and accumulated inference
timeouts before the protocol started; a separate 30-second reproduction failed
while waiting for the initial ordinary Responses headers. Models and capabilities
were available. The broad run was interrupted. The normal-timeout rerun reached
the protocol and produced the concrete results above, so the earlier timeouts
are not evidence of a masking bypass or a client dispatch failure.

## Reproduction and qualification

```bash
.venv/bin/pytest -n 0 --run-live 'tests/live/test_file_editing_live.py::test_live_model_edits_creates_and_removes_files[remove]'
.venv/bin/pytest -n 0 --run-live tests/live/test_file_editing_live.py::test_live_server_generates_an_exact_delete_marker
```

The isolated ordinary/marker-enum cases save the actual configure/resume bodies
and parsed raw JSON/SSE events to `delete-marker-wire.json` in their pytest temporary
directories, including IDs, schemas, snapshots, final arguments and interruptions.
Requests contain no credential headers. Keep the failing cases enabled under the
existing live opt-in; do not xfail them or count a crash/timeout as a successful edit.

In the 2026-10-09 qualification, live literal editing and new-file creation passed.
The 30-second real-model old_text
attack passed with three attempts, two complete independently validated calls and
three snapshot exchanges in 30.43 seconds in the final rerun; no attack call was dispatched and the
server/client were usable afterward. That result is bounded to the attempted
inputs and does not clear the deletion anomalies. Live deletion remains unqualified.

The 2026-10-09 deterministic client suite passed 2448 tests, with 47 opt-in tests
skipped; Ruff formatting/lint and Pyright passed. Mock SDK tests cover edit,
create and remove, invalid markers and selectors, changed files/paths, filtered
evidence, hooks, protocol IDs and retries. Additional regressions verify that
steering at rejection discards earlier completed calls, recovery stays in its
original skill, skill return precedes interruption replay, and interrupted
streams report final cumulative usage once. These checks qualify the client
guards without claiming that the live server anomalies have been fixed.

The 2026-10-11 clean-code review passes 2482 deterministic tests and repeats the
live editing checks: edit, create and the bounded old-text attack pass; deletion
and both isolated deletion-marker checks fail as described in S4. The
[review and combined live results](clean-code-review-2026-10-11.md) distinguish
these observations from a manual server restart, unsupported native attachments
and the macro workflow's context-window failure. No marker anomaly is cleared
by passing deterministic client guards.

## Server redesign audit: coded edit documents v2 (2026-10-11)

The server redesign is separate from the client implementation described above.
Before the change, a temporary C probe using the actual server recognizers
confirmed acceptance of all three truncated S2 replacement strings and the
ordinary S4 literal. For the diagnostic enum, a leading newline was accepted
as a prefix while the resulting framed value could not close. This establishes
a prefix/closure inconsistency, not the complete root cause of the recorded
budget failure.

V2 replaces markers with a single document string. Each edit path pauses before
its server-generated six-digit code attribute. The server prefills that attribute
after the snapshot reply, and the new_text closing delimiter must carry the
same code. This is an XML-inspired dialect, not standard XML. A bare
`</new_text>` in new_text is literal; CDATA can contain even the exact coded
delimiter. Unknown or truncated command tags cannot become ordinary commands.
Finite enum/const document schemas are rejected before generation because they
cannot describe the server-owned random attribute. The emitted normal Responses
metadata is also checked against the actual installed SDK.

Deterministic server regressions cover code ownership/matching, truncated
commands, newlines, native delimiter and UTF-8 fragmentation, CDATA/entities,
global selection ambiguity, sampling frontiers, repeated paths, rejection at
the second element, retries, cancellation and final frozen-resource validation.
The XML suite, ordinary server suite and ASan/UBSan passed. A server-produced
Response passed OpenAI SDK validation.

S1/S4 remain intent-fulfillment diagnostics: valid literal rewriting instead of
deletion is not a masking bypass. V2 does not infer deletion from similar text,
repair delimiters or promise to prevent every model choice. Numeric range
semantics, filesystem evidence, aliases, overlapping edits and writes remain
client responsibilities, even when invalid ranges pass server generation.

Live CUDA/Qwen27 testing found an additional server prefix defect: partial
all/line commands were pruned for empty/create snapshots even though their
complete forms passed. This forced the model toward range syntax. The server
now accepts those prefixes byte by byte, with permanent regressions. A later
MTP run reached a no-valid-token failure; review identified impossible native
escape continuations in structural/coded delimiters and at snapshot EOF, plus
old_text entity prefixes that could not match the next source scalar. These
are now rejected before commitment. Invalid UTF-8 and completed non-XML scalar
prefixes are rejected in paths and bodies, including CDATA.

The recorded v2 HTTP matrix runs 13 tests with two failing deletion subcases
(existing empty and nonempty files), no errors and no skips. All other cases
pass, including creation, streaming with a literal bare new_text close, line
insertion, deliberately invalid client-owned ranges, multiple elements,
repeated snapshots, rejection, retries and cancellation. The bounded attack
checks two completed constrained calls across three attempts; its one timeout
at the experiment deadline is recorded separately and is not a successful call.
The final scalar-hardening build repeats creation, streaming/collision and
multiple elements successfully.

Full-vocabulary optimized-mask/oracle comparisons report 486 checks with zero
divergences in the matrix and another 128 with zero divergences in the final
smoke run. MTP is active (50 accepted drafts in the matrix). These are mask
comparisons, not proof of target/MTP numerical inference equivalence.

Deletion still produces valid Python replacement text instead of the command.
At the first new_text frontier the unmasked argmax is token 198 (newline),
which is already allowed; a delete operation probe is also allowed. The
mask therefore does not exclude delete or force a different first token in
this observation. This does not identify the complete model/KV root cause.
Keep the live deletion assertions failing and do not rewrite the result.

The [v2 wire contract](../../docs/agentic/interactive-calls.md),
[qualification with profiling](../../docs/performance/interactive-edit-xml-v2-qualification-20261011.json)
and [actual synthetic request/response bodies](../../docs/performance/interactive-edit-xml-v2-wire-20261011.json)
record the server evidence. The temporary v2 instance was stopped and its
continuations/checkpoints released. Configurations and interruption replay
accept v2 only. The client migration remains pending; these server-only tests
do not qualify filesystem dispatch or end-to-end deletion.

The [complete test report](../../docs/performance/interactive-edit-xml-v2-full-tests-20261011.md)
adds all locally available library/agent live suites and target/MTP/batch XML
repetitions. Failures and skips are retained in the linked JSON, including
resource-limited and incorrectly configured MTP attempts.

The batch stream found a server prefix-viability defect: unfinished XML markup
followed by a partial native tool-delimiter escape could leave the decoder
with no valid token. The server now rejects that fallback before commitment,
including unfinished entities and UTF-8 scalars; native regressions cover the
actual DSML prefix as well as Qwen transport. ASan/UBSan and make test pass.

The library live run records 30 passes, 16 failures and one skip. Nine failures
request the obsolete v1 negotiation contract, six send native file content
instead of text/images, and one exceeds server context capacity. These belong
to client protocol migration, document conversion and history sizing. The
skipped browser bootstrap was run separately and passed. Agent live semantic
recovery checks still have three missing-feedback failures; malformed saved
output IDs/placeholders depend on client state and are not XML mask checks.
The portable typed SDK matrix passed all 20 cases on Qwen27; the actual Flash
model remains unverified.

Deletion and exact-coded-close CDATA intent assertions remain live failures
when the model chooses valid alternative content. A vision resident-cache
fixture also failed at a text continuation before its forced eviction checks;
server cache routing/replay remains unqualified by that run. The positive
image-conditioned tool-result 409/reuse test passed. None of these failures
is silently converted to a pass or excluded from the evidence.

After the prefix correction, the streaming/restore/collision case passes in
target, active MTP and four-session batch. Each full 16-method XML run retains
three failed intent assertions (CDATA and two deletion subcases). MTP also
records a 300-second creation timeout while repeating bare new_text closes as
literal content. The three phase profiles have 601/919/598 mask comparisons
respectively, zero divergences and no no-valid-token errors; MTP accepts 58
constrained drafts. Those facts do not turn the intent failures into passes.

On 12 October the default v2 framing removes code and its server prefill.
The model resumes after authorization by emitting >. Each edit ends with
the final </new_text> immediately preceding </edit>; earlier bare new_text
closes remain literal content. A literal complete </new_text></edit> sequence
still needs CDATA or XML entities. The client parser must use this rule per
element, preserving resource authorization and filesystem validation. Reports
dated 11 October describe the former coded framing and are historical evidence.

The 12 October clean-code review moves snapshot ownership into the document
and keeps suffix-index details behind matching APIs. The runtime consumes one
sampling policy and no longer carries the obsolete edit prefill callback.
Its complete live run exposed another server prefix defect: on a snapshot
containing aaa, the decoder admitted aa<![CDATA[]]>]] and then had no valid
token. Pending ]/]] bytes outside CDATA were omitted from logical old_text
matching. They are now checked against the snapshot before commitment;
native regressions also retain valid literal bracket selections.

The [review](../../docs/architecture/clean-code-review-20261012.md) and
[test report](../../docs/performance/interactive-edit-clean-review-20261012.json)
record the results and retained failures. Client protocol migration and
filesystem policy remain separate responsibilities.

## Client XML v2 follow-up — 12 October

The client now implements the code-free XML v2 contract: one document argument,
per-edit authorization and shared initial snapshots, post-hook document validation,
canonical aliases, numeric ranges/conflicts and filesystem commits. Legacy marker
commands and sanitation have been removed. They no longer appear in model-facing
tool/skill instructions or live generation tests.

Target-only client qualification initially recorded 7 passes and 7 failures. After
explicit application instructions and deterministic sampling it recorded 11 passes
and 3 failures, including the now-retired legacy-string case. The remaining deletion
and empty-file failures concern generated operation intent: valid rewrites or
recovery loops rather than a client parser/masking escape. The report keeps both
attempts, calls and exact file bytes. Final results and qualification limits are in
the [client review](../../docs/architecture/client-xml-edit-v2-20261012.md).

Final code-free client run: 11 passed, 2 failed out of 13 current cases; legacy
marker generation is absent. Deletion and empty-file intent remain failing.
MTP creation/alias/rejection qualification passes all four cases, with 15 actual
MTP cycles and five accepted draft tokens. Offline client tests pass 2,588 cases
with 100% statement/branch coverage; the frontend passes 938 tests including Playwright.
