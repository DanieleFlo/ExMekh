# Interactive text editing

`file-processing` includes one synchronous `edit-file(path, old_text, new_text)`.
All three arguments are required strings. It uses the same workroot, scope,
Before/After hooks, execution events, XML rendering and output cache as other
tools. It has no background arguments. Default selections and overrides must
include `edit-file` along with the other tools declared by this skill.

## Prerequisites and exact content

Existing files require a successful inspection followed by a successful text
read of the same version. Their actual calls and full, unchanged results must
remain visible in the current history. A selected read range qualifies; a
summary, filtered result, saved reference, failed read, imported history or
closed skill does not. Intervening unrelated tools are allowed. An inspected
empty text file can be read with `start_line=0, end_line=0`, returning zero lines.

The runtime stores node-local inspection/read provenance, canonical paths and
XXH3-128 hashes. It checks the current snapshot before authorizing the server,
and checks visible evidence, exact bytes and filesystem identity again after
Before hooks and immediately before committing. Missing inspection rejects the
incomplete call and appends a real synthetic inspection before the next ordinary
inference. Missing reading produces a distinct rejection without an automatic
read or another inspection. The model must make a new edit call after recovery.

Missing files use a separate creation authorization, without inspect/read.
The parent must exist inside the workroot. Permission errors, dangling symlinks,
directories, non-text formats and limits are errors, rather than absence.
Supported text is decoded using the existing file services without modifying,
normalizing or decorating the source. NUL and unsupported binary text are
rejected. The provided snapshot is exact Unicode text, without line numbers.

## Selections

The spellings below are the shipped defaults. `FileEditingSettings.marker_prefix`
and the three marker names are the sole source of marker syntax. Complete values,
the reserved family, escaped regexes, descriptions, errors and protocol metadata
are derived from them. Capability Markdown resolves explicit
`{{config.FILE_EDITING.all_marker}}` references when loaded, so the model receives
the configured values. A spelling change must also be accepted by the server;
client tests with alternate spellings use the SDK mock transport.

| Operation | `old_text` | `new_text` |
| --- | --- | --- |
| Replace a block | Exact nonempty block starting at a physical line boundary | Exact replacement, possibly longer or empty |
| Insert before line N | `@ex_meckh_line[N]` | Exact inserted text |
| Rewrite or create | `@ex_meckh_all` | Exact complete content, possibly empty |
| Create | `@ex_meckh_line[1]` | Exact complete content, possibly empty |
| Remove an existing file | `@ex_meckh_all` | `@ex_meckh_delete` |

Literal matching counts overlapping and mid-line occurrences globally. Exactly
one occurrence is required, starting before the line's indentation. A block may
span lines and finish mid-line. CR, LF and CRLF are preserved literally.
Insertion permits 1..L+1, with L+1 appending; an empty file permits only 1.
A final newline does not invent another physical line. No newline is added.

The entire `@ex_meckh_` family is reserved, including appearances in ordinary
source text and the incomplete bare stem `@ex_meckh`. Neither an incomplete marker
nor a newline-prefixed marker is repaired into a valid argument. The bare stem is
rejected even when the server completes it as ordinary replacement text.
Markers must occupy the entire correct argument. Embedded, repeated,
unknown, incomplete or wrong-field markers fail without a mutation. There is no
implicit escape syntax. `@ex_meckh_replace[...]` is a separate pending TODO and
is not supported by the v1 contract.

Provider arguments remain JSON:

```json
{"path":"notes.txt","old_text":"  old value","new_text":"  longer new value"}
{"path":"notes.txt","old_text":"@ex_meckh_line[2]","new_text":"inserted\r\n"}
{"path":"new.txt","old_text":"@ex_meckh_all","new_text":"created without newline"}
{"path":"notes.txt","old_text":"@ex_meckh_all","new_text":"rewritten\n"}
{"path":"notes.txt","old_text":"@ex_meckh_all","new_text":"@ex_meckh_delete"}
```

Results are structured Python values rendered with the shared XML renderer.
A successful result identifies `operation=edit`, `create` or `remove` and the
resulting byte count. Hook-adjusted arguments are stored in the execution cache;
the original provider call remains unchanged for history replay.

## DS4 transport v1

The authoritative server wire is
[`interactive-calls.md`](../../docs/agentic/interactive-calls.md). `LocalProvider`
uses its configured OpenAI client for ordinary requests and a dedicated custom
JSON/SSE adapter for scoped interactive tools. This requires `/v1/responses`, SDK
3.3 or later, negotiated v1 support, a supported loaded model syntax and the
`edit_file.v1` behavior. Chat Completions and structured-response requests reject
interactive tools explicitly. A server that ignores extra fields cannot qualify.

Tool definitions carry an `InteractiveDefinition` with the behavior, role-to-field
mapping, ordered `first_fields` and marker family. The provider configures only
tools in the advertised scope, using their actual registered names. Ordinary
JSON schema string types remain unchanged. Configuration is nested inside
`extra_body.agentic.interactive`, preserving context identity, scope and a pending
skill return. Control operations remove the outer return operation.

```text
ordinary Responses request + agentic.interactive configure
  -> accepted, then await_input after all priority values
runtime checks path, history and bounded snapshot
resume with only agentic (stream inherited)
  -> complete original call ID, or a terminal interruption
runtime dispatches complete calls and runs hooks once
  -> normal XML result, then the next ordinary inference
```

Each configure/resume/cancel has a stable request ID for retries. Continuation,
exchange and call IDs are checked separately, with frozen priority values and
schema types. A partial call never runs hooks or tools. No file content is added
to history during suspension; the out-of-context snapshot controls server masking.
The client independently checks every generated selection before filesystem effects.
The server owns logits masking, greedy selection, MTP gates, KV checkpoints and
the selection/replacement sampling frontier; the client does not implement them.

Rejection is an `InterruptedMessage`, distinct from an executable function call.
History export and cache preserve its structured record. Responses replay sends
the server's `interactive.rejected/cancelled/expired/failed` item, without invented
old/new arguments or orphan function outputs. Complete calls preceding a rejection
are processed separately; cancellation drops accumulated work. Original outcomes
precede any synthetic inspections. Steering or changed history cancels authorization.

Lost connections retry the identical payload/ID. Consumed continuations can be
recovered through status receipts, without a new ordinary inference. Unknown or
expired server state fails; it never authorizes a fresh application automatically.
The logical timeout and exchange budget span all resumes. SSE/JSON responses are
bounded in bytes, and HTTP handles close on success or error. Final cumulative
usage is used once; pause usage is not added. Server-preserved delta cursors keep
stream prefixes from repeating across resumes.

Some DS4 responses omit three SDK request-echo fields and the cache-write counter
required by SDK 3.3. The adapter supplies defaults only for these unused metadata
fields, then performs normal SDK response validation. Interactive IDs, calls and raw JSON argument
types are separately validated without coercion, including duplicate-key checks.

## Filesystem limits

`config/file_editing.py` configures 1 MiB snapshots/result files and generated
replacement text, a 300-second logical timeout, 64 exchanges and a 16 MiB wire
response budget. The server's smaller negotiated snapshot limit takes precedence.
No truncated content is sent or written. New files use UTF-8 without BOM;
existing BOM/byte order, encoding, permissions and untouched newline sequences
are preserved. Unrepresentable replacement text fails before committing.

Edits serialize on a cooperative canonical-path lock. A prepared same-directory
file is fsynced and atomically replaced. Creation publishes complete bytes through
an exclusive hard link, refusing a concurrently appearing target. Removal and
commit use a directory descriptor, preventing path redirection through a changed
parent. A platform without descriptor-relative operations fails explicitly.
The current qualification is Linux/WSL; native Windows commits are unsupported.

An exact-byte/identity check followed by rename or unlink is not an external
compare-and-swap. Processes that ignore these cooperative locks can still race
in the remaining check/commit window. There is no transaction across a batch:
after its first edit changes a file, another edit of that old snapshot fails.
Continuations and private evidence do not survive runtime/server restarts.

## Verification

Deterministic tests use real files and the actual OpenAI SDK with a mock HTTP
transport. They cover streaming/non-streaming edits, new/remove, empty reads,
literal uniqueness, markers, paths, hooks, filtered evidence, concurrent mutation,
identity changes, atomic/exclusive failure paths, encoding/BOM, limits, negotiation,
frozen renamed/multiple fields, escaped/fragmented JSON/SSE, rejection/replay,
retry/status receipts, steering and logical timeouts.

```bash
.venv/bin/pytest -n 0 tests/core/file_inspection/test_edit*.py tests/providers/test_interactive.py
.venv/bin/pytest -n 0 --run-live tests/live/test_file_editing_live.py
```

The live file tests use independent contexts and temporary files. The adversarial
test runs for 30 seconds with a real model, requesting invalid boundaries,
ambiguous/overlapping selections, marker indices, JSON types and delimiter escapes.
Every complete generated selection is checked by the independent client validator;
no attack call is dispatched to a filesystem. No attempted or completed constraint
checks means failure, and client crashes or unrelated transport errors do not count
as success. The test checks client/server availability afterward and records the
number of attempts, validated calls and snapshot exchanges.

Current live deletion failures and their possible masking/framing causes are
tracked in [server observations](interactive-edit-server-anomalies.md), with
captured evidence and isolated reproductions. They remain failing live tests;
the client does not normalize their invalid marker output into a deletion.
