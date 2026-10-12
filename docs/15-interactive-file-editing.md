# Interactive XML text editing

The synchronous `edit-file(edits)` tool accepts exactly one string containing
one or more consecutive XML fragments. Provider arguments and exported history
remain JSON; structured results use the shared XML renderer. The tool follows
the ordinary workroot, scope, Before/After hook, event and cache lifecycle, once
for the whole call, and cannot run in the background.

```json
{"edits":"<edit path=\"notes.txt\"><old_text>  old value</old_text><new_text>  updated value</new_text></edit>"}
```

## Grammar and exact content

The grammar is fixed by the coordinated v2 contract, without v1 fallback,
code attributes or server prefill. Each fragment contains path, old_text and
new_text in that order. ASCII framing whitespace is allowed between fragments
and between the final new_text close and edit close. Bodies retain CR, LF,
CRLF, indentation and trailing spaces exactly, without ordinary XML newline
normalization. Attributes preserve exact decoded filenames.

| Operation | old_text body | new_text body |
| --- | --- | --- |
| Replace | Nonempty unique literal from a physical line boundary | Exact replacement |
| Insert before physical line N | `<line number="N"/>` | Exact inserted text, with intended newlines |
| Rewrite/create | `<all/>` | Complete content, possibly empty |
| Create | `<line number="1"/>` | Complete content, possibly empty |
| Delete existing file | `<all/>` | `<delete/>` |
| Very long selection | `<old_text start_line="N" end_line="M"></old_text>` | Exact replacement |

Commands occupy the entire raw body. Entity-escaped or CDATA-wrapped commands
are literal text. The old `@ex_meckh_` strings are ordinary content, including
in files inspected or read; no marker sanitation or XML tag stripping occurs.

Literal matching counts overlapping and mid-line occurrences globally.
Exactly one occurrence is required, starting before indentation. A literal
may span lines and finish mid-line. Insertions permit 1..L+1; L+1 appends,
and an empty file permits only 1. A final newline adds no physical line.

Range attributes are digit strings at the protocol boundary. The client checks
positive values, ordering and existing inclusive physical lines, without
unbounded integer conversion. Range syntax is intended only for extremely long
selections. Error strings and limits are configured in `config/file_editing.py`.

The final plain `</new_text>` immediately before the fragment's `</edit>`
closes its replacement. Earlier bare `</new_text>` remains literal. A literal
complete `</new_text></edit>` pair requires entities or CDATA. This rule is
applied separately to each fragment, never to the last tag of the whole call.
Unknown markup, attributes, entities, DTDs, incomplete tags and invalid XML
characters fail without repair. Empty new_text removes selected content while
keeping the file; only an explicit deletion command removes it.

## Client authorization and filesystem ownership

Existing files require an actual successful inspection and subsequent text
read of unchanged content, both fully visible in the current history. Selected
read ranges qualify; summaries, filtered outputs, references, failed reads,
imported history and closed skills do not. Intervening tools are allowed.
An inspected empty file can be read using start_line=0, end_line=0.

The runtime owns canonical paths, node-local provenance and exact-byte hashes.
It authorizes every element separately using (call ID, edit index). Repeated
paths and symlink aliases reuse the same initial snapshot. Rejection of any
resource drops the entire XML call, including its earlier elements. Other
complete calls in the logical turn survive; cancellation discards accumulated
work. Rejections remain non-executable runtime records, with edit_index,
exchange ID, resource values and reason, and replay without invented arguments
or orphan outputs.

Missing inspection triggers a real synthetic inspection after the original
rejection and before the next ordinary inference. Missing reading requires
an explicit read followed by a new edit call. Missing files need neither.
Parents must already exist inside the workroot. Access errors, dangling
symlinks, directories, binary text and unsupported formats are resource errors,
rather than evidence of absence. Snapshots contain exact Unicode without line
numbers; unsupported NUL text is rejected.

After Before hooks, the client parses the final complete document, freezes its
path/index correspondence, groups aliases, checks visible evidence, ranges and
conflicts, encodes every result, and validates all targets before the first
write. Overlapping replacements, insertions inside a replacement and two
insertions at one position fail. Insertions at a replacement boundary are
allowed. Whole-file rewrite or deletion must be that file's only edit.

Each canonical file is committed once. Existing encoding/BOM, byte order,
permissions and untouched newline sequences are preserved. New files use
exact UTF-8 without an added newline. Result files and decoded snapshots are
bounded to 1 MiB; the complete generated document is bounded to 1 MiB.
Cancellation/scope changes are rechecked after hooks.

Commits serialize on cooperative canonical-path locks, acquired in stable order.
Each target's identity, exact bytes, canonical path and permissions are
rechecked. Same-directory temporary files are fsynced and atomically published;
creation uses an exclusive hard link. Directory descriptors protect against
parent redirection. Native Windows commits remain unsupported; Linux/WSL is
the qualified runtime. Non-cooperating external writers can still race in the
final check/commit window.

Distinct files have no transactional rollback guarantee. Invalid arguments or
known stale resources fail before all writes. A later operating-system failure
can leave earlier files committed; the structured error reports
`error_kind=partial_file_edit` and the completed `files` entries.

```python
{"ok": True, "files": [
    {"path": "a.txt", "operation": "edit", "size_bytes": 12},
    {"path": "b.txt", "operation": "create", "size_bytes": 0},
]}
```

## Server ownership and SDK transport

The [server wire contract](../../docs/agentic/interactive-calls.md) defines
version 2 and behavior `edit_file.v2`. The tool schema is one required string;
`InteractiveDefinition("edit_file.v2", {"document": "edits"})` can map the
document role to another registered argument name. It has no first_fields,
resource/selection/replacement roles or marker metadata.

LocalProvider uses its configured public OpenAI SDK client and the dedicated
custom JSON/SSE adapter. The provider negotiates capabilities, configures only
scoped tools and preserves context identity and pending skill return.
Chat Completions and structured-response calls reject interactive tools.

The server pauses before `>` after each decoded path, returning a fresh exchange
ID and consecutive edit_index. The runtime supplies provide/create/reject;
resume sends only agentic plus inherited streaming mode, without new history
or sampling. The server owns masking, parser/frontiers, sampling/MTP, KV,
checkpoints and continuation receipts. It does not access client files, check
permissions/evidence/ranges/conflicts, or apply mutations.

Stable request IDs, identical retry payloads, consumed-continuation status
receipts, steering and terminal validation remain in the transport layer.
The timeout and 64-exchange budget span the logical call. JSON duplicate keys,
coercion, changed paths, skipped indices and consumed exchange IDs are rejected.
SSE is bounded and decoded incrementally; handles close on every outcome.
Cumulative usage is reported once.

## Verification

```bash
.venv/bin/pytest -n 0 tests/core/file_inspection tests/providers/test_interactive.py
.venv/bin/pytest -n 0 --run-live tests/live/test_file_editing_live.py
```

Deterministic SDK tests cover multi-file and alias composition, body framing,
CDATA/entities, literal ambiguity, numeric ranges, conflicts, whole-call
rejection, preservation of prior calls, post-hook validation, cancellation,
encoding/BOM, atomic failures, evidence, negotiation, retries and replay.

Live tests exercise seven physical file operations, multi-file and alias calls,
a 30-second literal-selection attack, and second-resource rejection in both
streaming modes. They use deterministic sampling and assert exact filesystem
bytes. A valid but unintended model rewrite remains an application failure;
neither client nor server infers deletion intent or repairs generated content.
The [dated qualification report](../../docs/architecture/client-xml-edit-v2-20261012.md)
records current results separately from [historical server anomalies](interactive-edit-server-anomalies.md).

Legacy marker strings are no longer part of tool/skill instructions or live generation cases.
A deterministic literal-content regression verifies that the parser has no legacy command semantics.
