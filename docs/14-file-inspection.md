# Workspace file inspection

`inspect-file` analyzes one regular file inside the node workroot. It returns
metadata and a bounded navigation map, without returning the full content.
The entrypoint uses `@deep_tool`; filesystem access, parsing and result selection
live in separate modules under `core/file_inspection`. The snapshot owns
filesystem metadata; the navigation index owns its budget, ranking and pagination,
and exposes immutable selected pages rather than a mutable region list.

## Availability and calls

The `file-processing` skill and its `inspect-file` tool belong to
`DefaultCapabilities(harness=True)`. The bundled main HDS declares the skill
in `sub-skills`; `inspect-file` is available inside that skill, with parent
tools excluded. Custom HDSs must declare `file-processing` in `sub-skills`
to enter it. The skill currently provides analysis only. Individual selection,
exclusion, named overrides and Before/After hooks follow the existing
default-capability API.

Model arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `path` | required | Relative or absolute file path inside the workroot |
| `detailed` | `false` | Include filesystem metadata and configured limits |
| `max_items` | `50` | Regions per page, from 1 to 500 |
| `offset` | `0` | Zero-based offset in the selected navigation index |
| `query` | `null` | Rank the bounded index by lexical overlap with names/kinds |

Examples of provider tool-call arguments (the provider protocol remains JSON):

```json
{"path": "src/service.py", "query": "decode user message", "max_items": 10}
{"path": "docs/design.md", "offset": 50, "max_items": 50, "detailed": true}
{"path": "report.pdf"}
```

The runtime also exposes the common `execution_mode` and `wait_seconds`.
Background work uses `control_tool_execution` and `read_tool_output` unchanged.
Results are ordinary structured Python dictionaries; the shared renderer converts
them to typed XML at the presentation boundary. Immediate and saved/background
reads use the same rendering. No independent file-analysis cache is created:
the harness stores tool outputs according to its existing cache policy.

## Analysis

| Format | Result |
| --- | --- |
| Python / stubs | AST class and function index, qualified nested names, ranges including decorators; coding cookies supported; code is never executed |
| Markdown | Nested section ranges, ATX/Setext headings, backtick/tilde fences; headings inside fences ignored |
| JSON / YAML | Key paths and inclusive physical line ranges, node counts; values omitted; aliases bounded and cycles reported |
| TOML | Parsed key paths; line ranges omitted because the parser does not provide them |
| XML / SVG | Element names and line ranges; document types and entities rejected |
| CSV / TSV | Record/column counts, inconsistent widths, delimiter and heuristic header detection; navigation groups 100 records and respects multiline fields |
| ZIP / Office OOXML containers | Member names, counts, declared expanded size, encryption and unsafe-path indicators; no extraction or workbook/document interpretation |
| TAR, gzip/bzip2/xz TAR | Bounded member scan; explicitly partial totals when a budget stops the scan |
| PDF | Page count, encryption, document properties and first-page dimensions in points; no text extraction, rendering or OCR |
| Raster images | Format, dimensions, color mode and animation indicator from headers; no pixel payload, OCR or EXIF export |
| Other Unicode text | Encoding, physical line count, non-empty lines, maximum line length, newline style and whole-file navigation region |

UTF-8 and BOM-marked UTF-16/UTF-32 are supported. Unknown binary/non-Unicode
files return an explicit unsupported status. Other source languages receive
generic text analysis. PDF metadata uses the declared `pypdf` dependency;
encrypted PDFs report encryption and unavailable properties without requesting
passwords or attempting decryption.

Line ranges are inclusive and 1-based. A trailing newline does not invent an
extra line; CR, LF and CRLF count as physical line breaks, while Unicode text
separators do not. Structured parser positions are translated to these same
physical lines, including JSON/YAML strings with Unicode separators. Non-text
files do not claim a physical line count.

`query` ranks the complete *bounded* index before pagination. It tokenizes
camelCase/snake_case names and sorts by shared query terms, keeping the original
order for ties. It does not filter out nonmatches or download semantic models.
The region budget retains regions in document discovery order before ranking
and pagination. An `offset` applies to that ranked index; keep the same query
when paging.

## Status and resource limits

`analysis_status` is `complete`, `invalid_format`, `unsupported`, or
`skipped_limit`. A valid inspection may report unavailable/partial analysis
while keeping `ok=true` and metadata. Path, selection and permission errors
return `ok=false` with `error_kind=invalid_file_inspection`. Unexpected internal
failures propagate to the standard runtime error handling. Parser dependencies
are loaded at tool execution, so a missing dependency does not break capability
catalog loading. Such failures use the shared `missing_dependency` result
also returned by calculation and web tools.

`selection` distinguishes pagination (`has_more` / `next_offset`) from index
truncation (`index_truncated`) and label truncation (`labels_truncated`).
`indexed_items` counts retained regions, not hypothetical omitted ones.
A TAR with exactly the allowed number of members is complete; its scan is
partial only when a further member is omitted or the expanded-size budget stops
the scan. Warnings explain parser failures and stopped scans; valid text metrics
remain available when a format parser rejects its input.

Budgets live in `config/file_inspection.py`: 8 MiB input, 10,000 navigation
regions, 200 characters per label, 1,000 query characters, 64 structural levels,
100,000 YAML node visits, 10,000 archive entries, 64 MiB decompressed TAR input,
2,000 PDF pages and 40 million image pixels. ZIP members are never decompressed.
Oversized files expose metadata only. Changes detected while reading a file
produce a retryable error; each new inspection reads a fresh snapshot.

Deterministic tests create actual files and cover parsers, metadata, limits,
pagination, invalid paths, symlinks, concurrent modification detection, provider
schemas, background retrieval and a complete skill open/inspect/exit cycle
followed by a saved XML read from the parent scope. They do not contact a model provider.
