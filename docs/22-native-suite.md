# Native DS4 images and embeddings

`Client.native_suite()` exposes the configured native DS4 service on the same
OpenAI transport, URL and credentials as `LocalProvider`. No client residency,
checkpoint or load/unload operations are needed. Providers without this
capability raise the configured unsupported-provider error.

```python
from pathlib import Path
from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.native_suite import EmbeddingItem, TextContent
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.providers.local import LocalProvider

with LocalProvider(model="Qwen3.8-27B-UD-Q4_K_S", base_url="http://localhost:18091/v1") as provider:
    client = Client(provider, SystemMessage("Help the user."))
    suite = client.native_suite()
    image = suite.upload(Path("reference.png"), kind="image")
    vectors = suite.embeddings([
        EmbeddingItem(content=[TextContent(text="A reference"), image])
    ], dimensions=256, task="similarity")
    result = suite.images("A red square", width=256, height=256, seed=17)
    Path("result.png").write_bytes(suite.content(result.data[0].ds4.file_id))
```

The only implemented identities are `Literal["Qwen/Qwen-Image-2.1-Turbo"]`
and `Literal["google/embeddinggemma-2"]`. Each operation has one available
model, so methods and tools expose no model selector. Image dimensions are
integer `width` and `height`; serialization to the standard `size` field stays
inside the transport adapter. `models()` returns typed native catalog entries;
the provider's chat model selection excludes non-chat capabilities.

`embeddings` accepts ordered `EmbeddingItem` objects or a sequence of strings.
Parts are `TextContent` and uploaded `MediaContent` with type image/video/audio;
video `audio` is `exclude` by default and can explicitly be `include`. One
vector is returned per item. Supported dimensions are 128/256/512/768; task is
raw/search_query/document/code_query/similarity, with optional title for document.
The DS4 extension declares version 1 and rejects context overflow rather than
silently truncating. Standard OpenAI text/token embeddings remain available
through `provider.client.embeddings.create`.

`images` supports ordered image references, prompt-only, image-only with an empty
prompt, or both. It exposes count, integer dimensions, PNG format,
auto/opaque/transparent background, seed and idempotency key. The server fixes
the Turbo recipe at eight steps/CFG 1. Width/height are multiples of 32 from
32 to 2048; count 1–8 is bounded by 4,194,304 total pixels. Unknown options are
rejected. An automatic per-call idempotency key protects transport retries;
pass a stable key yourself when retrying the same logical call later.
References preserve their supplied dimensions (`reference_resolution=0`) and
must already satisfy the native Qwen3-VL geometry plan, such as 256×256.
References requiring an implicit resize are rejected before compute.

## Registered tools and results

`generate-image` is a discoverable bundled capability. Its normal registration
and per-node selection use `DefaultCapabilities`. Embeddings are an internal
service used by web search and ranking, not a standalone model-facing tool.
Tool execution propagates the calling `Client.native_suite` factory, including
background workers. Direct standalone tools/ranking use the provider environment
URL/key settings when no calling context exists.
The calling provider owns the shared HTTP transport. Standalone operations
open and close their own transport, including on failure. `LocalProvider`
accepts a public `http_client` argument for transport instrumentation and owns
that injected client until provider close; callers need no private SDK access.

`generate-image` accepts an ordered `images: list[str]` of workspace paths and
all implemented image options. It saves original PNGs under
`.exmekh_lib/generated-media/` and returns structured file IDs, paths, seeds,
width, height, recipe and service metadata. `include_image_in_result` defaults
to True in `config/native_suite.py`: an `AttachedToolResult` carries previews
through the existing message attachment path. `preview_scale` must be in `(0,1]`
and changes previews only. False returns structured metadata without pixels.
Previews include a short context label identifying the completed tool result.
History exports preserve their actual pixels using the existing cache codec,
including closed skill histories. Compact message dumps used for identity and
wire mapping retain their previous shape. The frontend accepts both these
lossless snapshots and its previously saved attachment format.
Structured results use the shared XML renderer at the presentation boundary;
Python values, HTTP bodies and stored records retain their existing types.

Web text and image ranking now share native EmbeddingGemma 2 similarity vectors.
OpenVINO, Granite and SigLIP2 ranking inference have been removed from the
client; Transformers/PyTorch remain dependencies of other existing features.
The native backend does not reuse client embedding caches across endpoints.
Text and visual ranking share one stateless backend. Per-call caches do not
mix endpoint settings or stale page contents; explicit text caches remain
available to callers who own their lifetime. Visual fusion is named
`compute_multimodal_ranking`, independently of any retired local model.
The existing BM25/provider fallbacks remain when semantic service calls fail.

## Service lifetime and verification

DS4 file IDs are temporary (3600 seconds, renewed on compute use); generated
local PNGs survive that server TTL. Receipt replay is bounded to 600 seconds
within the current service process. Generation/editing returns only after the
server restores any parked LLM state. A restore failure is an error and leaves
server recovery state intact. API/UI clients do not distinguish residency plans.

Offline tests cover serialization, order, dimensions, no model selector,
uploads, preview scaling, multiple outputs, path constraints, tool registration
and ranking backend replacement. DS4 live acceptance covers the actual OpenAI
SDK, all four embedding modalities, generation/editing and an actual ExMekh
LLM → registered tool → continuation with cached-token reuse. Frontend tests
cover thumbnails, the preview dialog and persistence after browser reload.
The DS4 T17/T18 report records commands, source/wheel hashes and limits.
The follow-up `docs/performance/native-suite-image-e2e-20261022/README.md` in the
DS4 repository audits every requirement in the `Funzionalità extra` checklist.
Its opt-in browser test uses a real LocalProvider, real registered tool and GPU
generation, then verifies two previews, the dialog, reload and a second model
turn with restored image attachments. A separate live matrix covers all three
input modes, preview suppression/scaling, original bytes, retry and native web
ranking. These tests qualify integration, not the separate image quality gates.
The subsequent `docs/performance/native-suite-clean-review-20261022/README.md`
records the ownership/codec refactoring and removal of the embedding capability
after clarifying that native embeddings belong inside web search.
