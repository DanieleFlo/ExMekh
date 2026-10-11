# Configuration consistency audit

Reviewed on 2026-10-09 across all configuration modules, Python source, bundled
capability instructions and tests. Literal matches were checked for their actual
meaning; an identical value is not necessarily the same configuration concept.

## Corrected divergences

| Area | Previous divergence | Shared definition now used |
| --- | --- | --- |
| File editing | Marker values, regexes, creation line, messages, skill instructions and tests could disagree | `FILE_EDITING.marker_prefix`, marker names and derived properties |
| PDF instructions | Skill always advertised five pages and 120 DPI | `FILE_INSPECTION.max_pdf_render_pages`, `pdf_render_dpi` |
| Inspection description | Advertised 50 items regardless of the signature's default | `FILE_INSPECTION.default_max_items` |
| Date/time instructions | Tool and skill always advertised UTC | `DATE_TIME.default_timezone` |
| Harness guidance | Sub-agent name repeated as literal text | `RUNTIME.sub_agent_tool_name` |
| Live model fixtures | Nine fixtures repeated the default model instead of using provider settings | `DEFAULT_PROVIDER_SETTINGS.model` |
| Provider tests | Environment keys and fallback credentials repeated configured defaults | `DEFAULT_PROVIDER_SETTINGS` |
| Cache/tokenizer tests | Paths, filenames, glob patterns, download URL and timeout repeated defaults | `DEFAULT_STORAGE_SETTINGS`, `CACHE_FORMAT`, `QWEN_TOKENIZER`, `TokenizerSettings` |
| Crawler tests | Environment key, cache directory and expiry assumptions repeated outside web settings | `WEB_FORMAT`, `WEB_NAVIGATION.cache_ttl_seconds` |
| Capability expectations | History limits, summary threshold and output/control names repeated in expected prose | `RUNTIME` |
| Live execution helpers | Runtime tool names repeated in prompts, filtering and event handling | `RUNTIME` |

Markers occur literally only at their default prefix definition in production
source and executable tests. Invalid-marker tests derive their malformed values
from the current configuration too. Saved server traces and the anomaly report
retain the exact values actually observed; they are historical evidence.

## Capability configuration references

`{{config.SETTINGS.field}}` is resolved before Markdown parsing, for public scalar
values from the explicit `FILE_EDITING`, `FILE_INSPECTION`, `DATE_TIME` and `RUNTIME`
registry. There is no evaluation or arbitrary import. Unknown/non-scalar references
fail before model use. Other braces, capability references and output placeholders
retain their existing meaning. Settings are read when capabilities are loaded;
already loaded tools are not automatically reconfigured.

## Other literal matches reviewed

JSON Schema keys such as `required`, `description` and `default`, provider API
selectors, HTML `header`, time-unit alternatives and message field `tool_calls`
are external/schema/data vocabulary, even where their spellings equal a default
in a config. They do not refer to the configurable tool-choice, cache folder,
XML header or timezone/unit default and must not be redirected to those settings.
Explicit sample names and layouts in independent capability/parser fixtures are
also test inputs rather than implicit defaults. This audit does not introduce a
general facility to rename the capability filesystem layout or the v1 wire protocol.

## Regression coverage

Actual SDK/node tests use alternate marker prefixes, regex metacharacters and
changed names for all three markers, covering rewrite, insertion, both creation
forms and removal. Separate tests verify updated reserved-family/creation errors.
Changed PDF limits/DPI, inspection defaults, timezone/unit and sub-agent name are
checked against advertised prompts and signatures. Reference parsing also covers
unknown fields, non-scalar values and preservation of other brace syntaxes.

The existing file-inspection and interactive-provider suites also passed all 343
tests in a fresh process with the prefix and all three marker names changed before
test collection. This checks the original test cases against alternate settings,
in addition to the dedicated configuration regressions.

A second fresh-process run passed 291 cache, tokenizer, tool-call archive and web
tests with changed temporary/session/record folders, filenames, record suffix,
web cache layout, crawler environment key and a 3600-second cache TTL.

Final validation: 2465 tests passed, 47 live opt-in tests skipped; Ruff lint and
format checks, Pyright and `git diff --check` passed. This configuration audit did
not rerun the live server qualification or resolve the deletion anomalies recorded
in [the server report](interactive-edit-server-anomalies.md).
