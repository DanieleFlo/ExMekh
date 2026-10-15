# Literal file editing and separate creation/deletion

The file-processing skill exposes inspect-file, read-file, edit-file,
new-file and delete-file. Model-facing results retain the shared XML renderer;
tool arguments and exported history retain their JSON transport.

## Public tools

- `new-file(path, content)`: create an absent path exclusively. Its parent must
  exist inside the workroot. Content is required and must contain at least two
  Unicode characters, counted without trimming. No inspection/read evidence.
- `delete-file(paths)`: remove files and directories using a nonempty list of
  workroot-relative paths, for example `{"paths": ["notes.txt", "build"]}`.
  The path determines the resource type; directories and their contents are
  removed recursively without following nested symbolic links. Absolute paths,
  directly named symbolic links (including links to internal targets),
  the workroot itself, missing entries, special files and paths resolving outside
  the workroot are rejected. No inspection/read evidence or text decoding;
  binary and large files are supported. Paths are processed in order, continuing
  after failures. The response contains `ok` (true only if every deletion
  succeeds) and `results`, with one entry per input path containing `path`, `ok`
  and either deletion metadata or `error`, `error_kind`, `error_code` and `hint`.
  Earlier successful deletions are retained; retry only failed paths. Deleting
  a directory can partially remove its contents before a filesystem failure.
  `FileDeletionTarget.resolve(...).remove()` owns resource validation, locking,
  identity checks and descriptor-relative removal; the tool adapter owns the
  ordered batch and converts each failure into its independent result.
- `edit-file(edits)`: edit existing Unicode text after successful visible
  inspection and subsequent reading of unchanged content. Missing files are
  errors; they are never created through edit.

## Literal two-marker document

```text
<exmekh_edit path="notes.txt" id="427" />old text<exmekh_new_text id="427"/>new text
```

The server assigns each opening ID after snapshot authorization and forces it
into the generated prefix. The separator copies that ID. IDs have three ASCII
digits, are unique within a call and survive continuation/retry.
The parser accepts omitted IDs with unambiguous framing. Explicit matching
separator IDs take precedence over bare literal markers. Authorized snapshots
disambiguate competing candidate selections; unresolved ambiguity is an error.
Three digits do not eliminate every possible delimiter collision.

Bodies are exact literal Unicode, including markup, entity spellings, CDATA,
CR/LF and trailing whitespace. Only the path attribute is entity-decoded.
JSON/native transport escapes are independent of document framing.
Replacement ends at the end of the string or the next edit opening with a
fresh ID; no closing tags are added and inter-block whitespace belongs to
the previous replacement.

Old text must start at a physical line boundary including indentation and
occur exactly once globally, including overlapping and mid-line matches.
Only an existing empty file allows empty old text. Insert by retaining a
literal anchor in new text and adding before/after it. Empty new text removes
the selection and keeps the file. All, line, range and delete tags have no
command semantics.

## Runtime ownership

`InteractiveDefinition("edit_file.v3", {"document": "edits"})` negotiates the
new behavior over Responses transport version 2. There is no v2 document
fallback. Each path/index is authorized separately before body generation;
the runtime replies provide/reject, never create.

All edits refer to initial snapshots. Hooks cannot change frozen resource
correspondence. Evidence, aliases, uniqueness, conflicts, result encoding,
concurrency, cancellation and scope are rechecked before commits.
Each canonical file is written once. Existing encoding/BOM, permissions and
untouched newline bytes are preserved. New files contain exact UTF-8 without
an added newline and are published exclusively.

The generated document, decoded edit snapshot and result remain bounded to
1 MiB. Delete has no text snapshot limit. Directory descriptors and cooperative
locks protect filesystem publication. Linux/WSL is the qualified commit
runtime; non-cooperating writers can still race in the final check/unlink
window. Distinct files have no transactional rollback guarantee.

## Verification

```bash
.venv/bin/pytest -n 0 tests/core/file_inspection tests/providers/test_interactive.py
.venv/bin/pytest -n 0 --run-live tests/live/test_file_editing_live.py
```

The deterministic suite covers separate new/delete hooks, absent/existing
paths, binary and recursive directory deletion, independent batch failures,
workroot and symbolic-link boundaries, empty files, literal old command tags, omitted IDs,
snapshot-based separator disambiguation, exact content, aliases, post-hook
validation, cancellation, encoding/BOM and exclusive/atomic failures.
