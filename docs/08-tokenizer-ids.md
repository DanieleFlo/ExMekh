# Tokenizer-based IDs

`LocalProvider` accepts an optional `tokenizer` implementing
`exmekh_lib.interfaces.tokenizer.Tokenizer`. It validates the instance and defaults to
`QwenTokenizer`, using the vocabulary from `Qwen/Qwen3.8-27B` on Hugging Face.
Only `tokenizer.json` is downloaded, without model weights. The `tokenizers`
package loads the complete tokenizer for text operations.

```python
from exmekh_lib.providers.local import LocalProvider
from exmekh_lib.tokenizers.qwen import QwenTokenizer

tokenizer = QwenTokenizer()
tokenizer.initialize()  # Optional: prepare the cache at application startup.
provider = LocalProvider(model="your-server-model", tokenizer=tokenizer)
identifier = provider.generate_id()
# Equivalent: provider.get_tokenizer().generate_id()

text_tokenizer = provider.get_tokenizer()
count = text_tokenizer.count_tokens("some text")
prefix = text_tokenizer.truncate_text("some text", start_tokens=100)
ends = text_tokenizer.truncate_text("some text", start_tokens=50, end_tokens=50)
```

`count_tokens()` and `truncate_text()` encode text without automatically added
special tokens. Truncation preserves the original characters at each end and
returns the input unchanged when the requested portions cover it. It rejects
negative or non-integer token counts. The exit tool uses the first 100 tokens
of the configured tokenizer for its thinking in a deep skill's final XML header.

Initialization is otherwise deferred until the first ID request. The vocabulary
keeps only complete entries matching `[a-zA-Z0-9]+`, excludes special tokens,
deduplicates entries and sorts them deterministically. Spaces, punctuation,
Unicode letters, and tokenizer whitespace markers are rejected, never stripped.
An ID joins three vocabulary entries with underscores. This does not imply that
encoding the entire resulting string, including separators, takes three tokens.

For a vocabulary of N entries, a base-N counter produces N³ distinct IDs without
materializing combinations or keeping a growing list of issued IDs. Components
may repeat within an ID. Exhaustion raises `RuntimeError` rather than wrapping.

## Cache and session isolation

The default root is `.exmekh_lib/tokenizers` relative to the working directory when the
tokenizer is constructed. Pass `cache_directory=Path(...)` to choose another root.
Each tokenizer identity has its own SHA-256 namespace:

```text
.exmekh_lib/tokenizers/<identity-hash>/
    vocabulary.json
    tokenizer.json
    temp/session-<unique-suffix>/counter.json
```

The identity includes the model, revision and filtering version. Cache metadata
also records the identity. A valid existing cache is read without network access;
an invalid cache raises an explicit error instead of silently resetting state.
Writes use a temporary file and atomic replacement, so failures do not publish
partially written JSON.
Older caches containing only `vocabulary.json` remain valid for ID generation;
the first text operation downloads and caches `tokenizer.json`.

A session lasts for one Python process. Instances and providers sharing the same
identity, vocabulary fingerprint and resolved cache directory share a counter,
including concurrent threads. Different
tokenizer identities have separate namespaces and counters, even when their
vocabularies overlap. **Uniqueness is scoped to a tokenizer and process**, not
across different tokenizers or executions. Keep that namespace when using IDs
as keys or constructing node-specific paths.

Temporary session directories are removed automatically on normal process exit.
After a forced termination, abandoned directories may remain; subsequent runs
never resume or delete another session's state. A new process starts at zero
while reusing the vocabulary. Counter storage is constant-sized. If a write
fails, the reserved ID is skipped, ensuring it is not reissued in that session.

## Adding a tokenizer

Subclass `SequentialTokenizer` to reuse filtering, caching and ID allocation.
Choose a stable, distinct identity including any relevant configuration/revision,
and implement `_load_vocabulary_tokens()` to return raw, non-special vocabulary entries:

```python
from collections.abc import Iterable
from pathlib import Path

from exmekh_lib.core.tokenizers.sequential_tokenizer import SequentialTokenizer


class OtherTokenizer(SequentialTokenizer):
    def __init__(self, cache_directory: Path | None = None) -> None:
        super().__init__("vendor/model:revision:ascii-v1", cache_directory)

    def _load_vocabulary_tokens(self) -> Iterable[str]:
        return ["house", "yellow", "green"]  # Load the real vocabulary here.

    def count_tokens(self, text: str) -> int: ...  # Use this model's text tokenizer.

    def truncate_text(
        self, text: str, start_tokens: int, end_tokens: int = 0
    ) -> str: ...  # Preserve the requested portions of the original text.
```

Implement `Tokenizer` directly only when providing your own ID allocation logic.
Its public contract consists of `initialize()`, `generate_id()`,
`get_session() -> SessionCache`, `count_tokens()` and `truncate_text()`. The
session returned by `get_session()` is also used by the node runtime to save
tool outputs; a custom tokenizer must implement this method even if it uses a
different ID allocation strategy.

```python
session = provider.get_session()
# Equivalent: provider.get_tokenizer().get_session()
application_cache = session.node("application-owned-node")
```

The explicit node ID above addresses an application-owned cache. Runtime nodes
use generated IDs, so an HDS name alone does not identify their cache. See
[Tool output caching](09-tool-call-cache.md) for storage layout, lifecycle,
search, and standalone archives.

## Responsibilities and dependencies

- `Provider` receives a `Tokenizer` and calls only its public methods; it does not
  import or construct a concrete tokenizer. Custom providers must pass their
  tokenizer to `super().__init__(tokenizer)`.
- `LocalProvider` selects Qwen when no tokenizer is supplied. This is the point
  where the default implementation is composed with the provider.
- `SequentialTokenizer` orchestrates vocabulary preparation and ID generation.
- `VocabularyCache.load_or_build()` owns vocabulary filtering, validation and
  persistent caching.
- `TokenizerSessionRegistry.get_session()` delegates session sharing to
  `CacheManager` and its synchronized `SessionPool`.
- `TokenizerSession` is a compatibility alias for `SessionCache`.
  `SessionCache.reserve_next_index()` owns counter synchronization and
  temporary storage. Callers do not access its counter, lock or filesystem layout.
- `exmekh_lib.utils.file.write_json_atomically()` provides generic JSON persistence used
  independently by the vocabulary cache and the session counter.
