# Tool error guidance

Bundled non-mock tools distinguish failures when the recovery action differs.
The existing `error_kind` identifies the error family. Structured failures also
expose `error_code` for the detected cause, `error` for the diagnostic, `hint` for
the next action, and optional `details` for parameter names or observed state.
Dependency results already identify `missing_dependency` and keep their existing
installation hints and shape. The existing `file_inspection_required` prerequisite
result retains its shape and already describes the required inspection and new read.

Python results and cache records remain structured. The shared XML renderer adds
types, escapes text, and uses base64 for characters XML cannot represent. Immediate
results, background results and saved reads preserve the same diagnostic fields.
Provider arguments and history exports keep their JSON protocol.

## Detection and compatibility

`ToolDiagnosticError` carries a cause at the detection site. `ToolInputError` and
`ToolTypeError` retain compatibility with `ValueError` and `TypeError` respectively.
The file, math, date/time and web domain errors use the same diagnostic support.
`ToolArgumentError` remains distinct from errors raised by a tool body or hook.
Binding diagnostics identify missing/unexpected parameters and list required and
accepted names. Existing Python callers can still catch the original exception
families; consumers of result dictionaries must allow the additional fields.

Recovery text lives in `config/tool_errors.py`; domain diagnostic text remains in
its existing configuration module. New codes use explicit causes or exception
types/OS errno; existing provider/solver classification is preserved. Arbitrary internal
`TypeError` and `ValueError` remain execution failures. Custom diagnostic codes can
supply their own hint; unrecognized codes receive general execution-failure guidance
without assuming the input is wrong.

Diagnostic codes and hints are read-only properties. Details and rendered fields
are independent snapshots, including nested containers. Observed context is
recorded through `add_details`; callers do not mutate an error's internal state.
Binding diagnostics belong to `ToolArgumentError`, and HTTP classification belongs
to `WebNavigationError`. Recovery hints are exposed as a read-only mapping.

## Reviewed tool inventory

| Tools | Distinct causes and recovery |
| --- | --- |
| All work tools | Missing/unexpected arguments, invalid execution options, internal execution/hook failures. Correct arguments before invoking; inspect state before retrying work that may have had effects. |
| `inspect-file` | Invalid/outside/unresolvable path, missing file, non-regular resource, unavailable workroot, permissions, changed content and inspection parameters. Unsupported analysis and bounded scans retain successful metadata with warnings. |
| `read-file` | Missing/stale inspection, invalid or out-of-bounds indices, empty text, unsupported text, corrupt/unreadable image/PDF, page/byte/pixel limits. Automatic inspection remains separate from the rejected read; choose indices again from its result. |
| `edit-file` | Missing inspect/read evidence, unsupported text, file changes, authorization mismatch, invalid framing, byte limits, cancellation. Literal selection separately identifies empty, missing, ambiguous and mid-line old text, including overlapping occurrences. Overlapping selections remain distinct. |
| `new-file` | Existing destination, absent parent, invalid/minimum content, byte limits, filesystem errors. Never overwrite an existing path; select an unused path or inspect/read/edit existing text. |
| `delete-file` | Missing/non-regular resource, path changes, unavailable platform and filesystem failures. Deletion does not acquire inspect/read requirements. |
| `search-web` | Query/result-count/timeout validation, local image path/format limits, remote image discovery and provider search failures. Empty successful searches are not errors; unavailable optional ranking remains a fallback signal. |
| `read-web-page` | Invalid/blocked URL, DNS failure, unavailable browser, crawl timeout/failure, cleanup failure and navigation-policy rejection. Observed HTTP failures distinguish 404, 401/403, 429 and 5xx even when the crawler reports success. A crawl without a status retains the general cause; no HTTP status is inferred from text. |
| `interact-web-page` | Action schema rejection, action timeout/failure, unobserved effects and navigation-policy rejection. Outcomes keep the original error field, add diagnostic text/guidance, stop on action exceptions, and preserve earlier successful actions. A navigation rejection carries completed actions in details. An unobserved effect does not change the existing success flag. |
| `calculate` | Expression syntax/unsupported constructs, function argument counts, complexity/numeric limits, unbound evaluate variables, incomplete integration bounds, matrix shape/entries and singular inverses. Only `log` accepts an optional second argument for its base; other supported functions are unary. Undefined or non-finite numeric evaluations are rejected instead of returned as ordinary numeric results. A singular determinant of zero remains valid. |
| `logic-check`, `logic-consistency`, `logic-counterexample` | Formula parsing, undeclared variables, type mismatch, unsupported theory/operation, dependency/internal failures, timeout and resource limits. Unknown/timeout/resource-limit outcomes give guidance and remain inconclusive; valid counterexamples and inconsistent premises retain their existing result semantics. |
| `date-time` | Invalid ISO datetime/timezone/duration, ambiguous or nonexistent local time, range overflow and unavailable timezone database. Correct DST ambiguity with an explicit offset in the datetime. |
| `get_execution_history` | Invalid 1-based navigation and page limit. Empty/out-of-range pages remain navigable successful results. |
| `search_tool_outputs` | Invalid query/filter/path/offset/page limit. Empty results remain successful; omit filters to broaden a search. |
| `read_tool_output` | Missing output ID, invalid/unsupported cursor, invalid summary mode or absent summary provider, partial-read/export failures. The existing `found=false` XML result now explains how to locate the saved output ID. Saved running work is still unavailable after restart. |
| `control_tool_execution` | Unknown/ambiguous original call ID, invalid action/wait, cancellation failure. Launch also distinguishes active duplicate IDs and the active-work limit. Unsupported cancellation retains the current status rather than claiming the work stopped. |
| `bash` | Missing executable versus missing workroot, process/pipe failures and OS errors. Nonzero exit codes remain completed command results with stdout/stderr; they are not harness execution failures. |
| `call_sub_agent` | Invalid child selector, empty text and unexpected fields show available children and a corrected call using the registered tool name. Child/provider execution failures still propagate to their existing boundary. |
| `exit_agent` | Existing precise messages cover invalid text/items, malformed or unlisted placeholders, duplicate aliases, unknown/unretrievable IDs, ambiguous call IDs and outputs outside the current invocation. These messages already explain correct alternatives and are preserved. |

## Recovery after effects

`partial_file_edit` lists committed files and the failed path, retains the cause
code (for example `permission_denied`), and explicitly says to reread current
targets and submit only remaining changes. Validation before writing still rejects
the entire call; an OS failure between commits can leave earlier commits applied.

Browser action failures do not imply that earlier actions failed. A click can also
have an effect before a timeout or navigation rejection. Read the current page and
use the recorded outcomes before repeating submissions.

Later crawl, page-capture or cleanup failures retain the original action failure
and observed outcomes. Cleanup failures accompanying a primary failure are logged
and attached as context; a cleanup failure on its own has a separate cause code.
Inspection and editing share workroot validation, so an unavailable workspace
does not produce advice to create the requested file.

`tool_output_export_failed` describes a delivery failure. Its guidance explains
that tool work may have completed and directs recovery through the existing output
ID before relaunching the tool; the underlying diagnostic still distinguishes an
OS error such as a full disk. A failed partial read adds `partial_output_hint` to
an immediate running result or `partial-output-hint` to the read's XML header,
without advancing an invalid cursor or changing execution status.

## Intentional boundaries and format exceptions

- A domain result with `ok=false` remains a completed/final tool result. A raised
  execution error remains failed. Delivery failures retain their existing failed
  presentation metadata without pretending the underlying work was undone.
- Scope and unknown-capability checks still reject the batch before work starts.
  Their exception messages now identify allowed/registered names and the recovery
  path. They are not converted into individual tool results, preserving batch and
  history validation.
- Cache/storage and provider failures outside tool invocation still propagate;
  they are not reclassified as fixable model arguments.
- Only execution-option validation is classified as an invocation argument error.
  Manager launch failures, duplicate active calls and capacity limits retain the
  execution-failure family and their detected cause.
- Existing tool-authored exit/subagent text, preformatted results, `found=false`
  XML and interactive provider control messages retain their formats. They are
  intentional exceptions to the common structured failure dictionary.
- Summary-provider failure still falls back to the complete output. Optional
  ranking failures and file-analysis warnings remain warnings/fallbacks.

## Validation

Deterministic tests cover cause-specific results from real bundled tools, corrected
next calls, OS classification, XML round trips with invalid characters, unchanged
file contents after rejection, partial commits, action outcomes, HTTP status
classification, DST correction, singular/undefined mathematics and identical
inline/managed/saved diagnostics. Browser/provider behavior is tested with controlled
fixtures; live provider qualification remains a separate opt-in suite.

Validation on 2026-10-14: the complete offline suite passed 2874 tests, with 55
skipped tests and 100% statement and branch coverage. Ruff lint/format, Pyright
and diff whitespace checks passed; wheel and source builds succeeded offline.
Live tests were not run for this task. See the
[clean-code review](clean-code-review-error-guidance-2026-10-14.md) for the ownership,
encapsulation and argument-isolation checks.
