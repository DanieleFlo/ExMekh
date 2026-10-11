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
