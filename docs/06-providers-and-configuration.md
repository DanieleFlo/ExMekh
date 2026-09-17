# Providers & Configuration

The provider layer decouples `qs` from underlying LLM servers. It handles authentication, base URL resolution, API protocol translation, model availability validation, and request/response mapping.

---

## 1. `LocalProvider`

[`LocalProvider`](file:///c:/Users/danie/Desktop/qs/qs/src/qs/providers/local.py) is the primary provider for interacting with OpenAI-compatible LLM backends (such as llama.cpp, Ollama, vLLM, LM Studio, or OpenAI services).

### Initialization Parameters

```python
from qs.providers.local import LocalProvider
from qs.core.models.reasoning_effort import ReasoningEffort

provider = LocalProvider(
    model="Qwen3.8-27B-UD-Q4_K_S",
    base_url_env_var="BASE_URL",  # Env var name for base URL (default: "BASE_URL")
    api_key_env_var="API_KEY",  # Env var name for API key (default: "API_KEY")
    api="chat_completions",  # Endpoint style: "chat_completions" or "responses"
    reasoning_effort=ReasoningEffort.MEDIUM,  # Optional reasoning effort setting
)
```

### Model Availability Check
When `LocalProvider` initializes, it queries the backend's `/v1/models` endpoint. If the requested `model` name is not returned by the backend, it raises a `ValueError` listing available model names.

---

## 2. API Endpoint Protocols

`LocalProvider` supports two endpoint protocol modes:

### Chat Completions API (`api="chat_completions"`)
Targets standard `/v1/chat/completions` endpoints.
- Handled internally by [`ChatCompletionOpenAI`](file:///c:/Users/danie/Desktop/qs/qs/src/qs/core/providers/chat_completion_openai.py).
- Uses standard JSON payload formatting for tool calls (`tools`, `tool_choice`).
- Converts streamed chunks to `ProviderStreamEvent`.

### Responses API (`api="responses"`)
Targets the OpenAI `/v1/responses` endpoint syntax.
- Handled internally by [`ResponseOpenAI`](file:///c:/Users/danie/Desktop/qs/qs/src/qs/core/providers/response_openai.py).
- Supports specialized output modalities and response structures.

---

## 3. Environment Variables & `.env` Setup

`qs` integrates with `python-dotenv` to load configuration settings from a local `.env` file.

### Environment Variable Reference

| Variable | Required | Default Value | Description |
|---|---|---|---|
| `BASE_URL` | **Yes** | None | Base HTTP/HTTPS URL of the LLM provider (e.g. `http://localhost:8000/v1`) |
| `API_KEY` | No | `"local-api-key"` | API authentication key. If omitted, defaults to `"local-api-key"`. |
| `QS_LIVE_MODEL` | No | `"qwen3.5-9b-mtp"` | Target model name when running live test suites with `pytest --run-live`. |
| `QS_RUN_LIVE_TESTS` | No | `0` | Set to `1` to enable live tests without passing the `--run-live` pytest flag. |

### Sample `.env` File
```ini
BASE_URL=http://localhost:8080/v1
API_KEY=sk-local-secret-key
QS_LIVE_MODEL=qwen3.5-9b-mtp
```
