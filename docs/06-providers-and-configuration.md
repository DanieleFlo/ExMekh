# Providers & Configuration

The provider layer decouples `exmekh_lib` from underlying LLM servers. It handles authentication, base URL resolution, API protocol translation, model availability validation, and request/response mapping.

---

## 1. `LocalProvider`

[`LocalProvider`](../src/exmekh_lib/providers/local.py) is the primary provider for interacting with OpenAI-compatible LLM backends (such as llama.cpp, Ollama, vLLM, LM Studio, or OpenAI services).

### Initialization Parameters

```python
from exmekh_lib.providers.local import LocalProvider
from exmekh_lib.core.models.reasoning_effort import ReasoningEffort

provider = LocalProvider(
    model="Qwen3.8-27B-UD-Q4_K_S",
    base_url="http://localhost:8080/v1",
    api_key="local-api-key",
    api="chat_completions",  # Endpoint style: "chat_completions" or "responses"
    reasoning_effort=ReasoningEffort.MEDIUM,  # Optional reasoning effort setting
)
```

`base_url` and `api_key` are keyword-only and take precedence over environment
variables. If they are omitted, `base_url_env_var` and `api_key_env_var` select
the environment variable names (`BASE_URL` and `API_KEY` by default). Explicit
configuration is recommended when multiple providers may be created concurrently,
because it does not mutate or depend on process-global environment state.

### Model Availability Check
When `LocalProvider` initializes, it queries the backend's `/v1/models` endpoint. If the requested `model` name is not returned by the backend, it raises a `ValueError` listing available model names.

---

## 2. API Endpoint Protocols

`LocalProvider` supports two endpoint protocol modes:

### Chat Completions API (`api="chat_completions"`)
Targets standard `/v1/chat/completions` endpoints.
- Handled internally by [`ChatCompletionOpenAI`](../src/exmekh_lib/core/providers/chat_completion_openai.py).
- Uses standard JSON payload formatting for tool calls (`tools`, `tool_choice`).
- Converts streamed chunks to `ProviderStreamEvent`.

### Responses API (`api="responses"`)
Targets the OpenAI `/v1/responses` endpoint syntax.
- Handled internally by [`ResponseOpenAI`](../src/exmekh_lib/core/providers/response_openai.py).
- Supports specialized output modalities and response structures.

---

## 3. Configuration modules

Configuration lives in `src/exmekh_lib/config/` and is separated by responsibility:

| Module | Class | Purpose |
|---|---|---|
| `provider.py` | `ProviderSettings` | User-selectable CLI model and API, provider environment variable names, and local API key fallback |
| `storage.py` | `StorageSettings` | User-selectable tokenizer and session cache directory |
| `tokenizer.py` | `TokenizerSettings` | User-selectable tokenizer download timeout |
| `cli.py` | `CliSettings`, `CliPresentation` | User-selectable HDS names and CLI output text |
| `tokenizer.py` | `QwenTokenizerSpec` | Fixed repository, revision, vocabulary identity, and filename |
| `cache.py` | `CacheFormat` | Fixed cache file and folder names |
| `capabilities.py` | `CapabilityFormat`, `CapabilityText` | Fixed capability layout, metadata fields, and bundled descriptions |
| `validation.py` | `ValidationPatterns` | Fixed patterns for names, references, and cache paths |
| `runtime.py` | `RuntimeSettings` | Fixed harness tool names, limits, and instructions |
| `web.py` | `WebNavigationSettings` | Shared web tool defaults, timeouts, cache lifetime, and navigation, output, and image resource budgets |
| `web.py` | `WebRankingSettings` | Embedding model defaults, cache capacity, and BM25/RRF ranking weights |
| `web.py` | `WebFormat` | Fixed web cache layout, OpenVINO artifact filenames, image formats, MIME types, and user agent |
| `reasoning.py` | `MathSettings`, `FormalLogicSettings` | Shared calculation precision, logic timeouts, input budgets, and solver resource limits |

The settings classes are frozen. To customize an application, construct its own
settings instance or set the environment variables below before starting the CLI.
Runtime protocol values are kept separate because changing them independently
would break the corresponding tools and cached data. Explicit arguments to
`LocalProvider` and the tokenizer still take precedence over defaults.

Web and reasoning defaults are centralized in `WEB_NAVIGATION`, `WEB_RANKING`,
`MATH`, and `FORMAL_LOGIC`. Their bundled tool entrypoints use the same defaults
as the core implementations. Memory pagination defaults and limits are part of
`RUNTIME`. These modules do not read environment variables; the `EXMEKH_LIB_*` variables
below apply only to the settings that explicitly support environment loading.
Default function arguments and existing module-level constant aliases are bound
when their modules are imported; these constants are not a live reconfiguration API.

### Environment variables

The CLI loads a local `.env` file. The settings classes read the relevant
`EXMEKH_LIB_*` values. `LocalProvider` reads `BASE_URL` and `API_KEY`, while the live
test suite reads `EXMEKH_LIB_LIVE_MODEL` and `EXMEKH_LIB_RUN_LIVE_TESTS`.

| Variable | Default | Description |
|---|---|---|
| `EXMEKH_LIB_MODEL` | `Qwen3.8-27B-UD-Q4_K_S` | Model used by the CLI |
| `EXMEKH_LIB_API` | `responses` | CLI API protocol: `responses` or `chat_completions` |
| `BASE_URL` | None | Provider URL; required unless passed to `LocalProvider` |
| `API_KEY` | `local-api-key` | API key used by `LocalProvider` if no key is supplied |
| `EXMEKH_LIB_TOKENIZER_CACHE_DIR` | `.exmekh_lib/tokenizers` | Cache root for tokenizer and sessions |
| `EXMEKH_LIB_TOKENIZER_DOWNLOAD_TIMEOUT` | `60` | Positive integer timeout in seconds for tokenizer download |
| `EXMEKH_LIB_ROOT_HDS` | `main` | Root HDS used by the example CLI |
| `EXMEKH_LIB_CHILD_HDS` | unset | Optional child HDS for the CLI; no child is created by default |
| `EXMEKH_LIB_LIVE_MODEL` | `Qwen3.8-27B-UD-Q4_K_S` | Model used by live tests |
| `EXMEKH_LIB_RUN_LIVE_TESTS` | `0` | Set to `1` to enable live tests |

The CLI assigns an explicitly configured child the context ID
`<root-hds>/<child-hds>`, so it can reuse `main` without sharing the root context.

The HDS names must refer to capabilities available in the selected capability
paths. The cache directory can also be passed directly to a tokenizer or
`CacheManager`. Existing cache data is not moved when its location changes.

### Sample `.env`

```ini
BASE_URL=http://localhost:8080/v1
API_KEY=sk-local-secret-key
EXMEKH_LIB_MODEL=Qwen3.8-27B-UD-Q4_K_S
EXMEKH_LIB_API=responses
EXMEKH_LIB_TOKENIZER_CACHE_DIR=.exmekh_lib/tokenizers
EXMEKH_LIB_LIVE_MODEL=Qwen3.8-27B-UD-Q4_K_S
```
