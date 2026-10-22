"""Versioned DS4 media operations on the same OpenAI transport as chat."""

from __future__ import annotations

import io
import mimetypes
import os
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field

from exmekh_lib.config.native_suite import NATIVE_SUITE
from exmekh_lib.config.provider import DEFAULT_PROVIDER_SETTINGS


class TextContent(BaseModel):
    model_config = {"extra": "forbid"}
    type: Literal["text"] = "text"
    text: str = Field(min_length=1)


class MediaContent(BaseModel):
    model_config = {"extra": "forbid"}
    type: Literal["image", "video", "audio"]
    file_id: str
    audio: Literal["include", "exclude"] | None = None


class EmbeddingItem(BaseModel):
    model_config = {"extra": "forbid"}
    content: list[TextContent | MediaContent] = Field(min_length=1)


class GeneratedMedia(BaseModel):
    file_id: str
    seed: int
    width: int
    height: int
    recipe: dict[str, Any]


class GeneratedImage(BaseModel):
    b64_json: str
    ds4: GeneratedMedia


class ImageBatch(BaseModel):
    created: int
    data: list[GeneratedImage]
    ds4: dict[str, Any]


class NativeCapabilities(BaseModel):
    version: Literal[1]
    revision: str
    profile: Literal["bf16-native-v1"]
    operations: list[Literal["image_generate", "image_edit", "embed"]]
    input_modalities: list[Literal["text", "image", "video", "audio"]]
    configured: bool
    resident: bool
    ready: bool


class NativeModel(BaseModel):
    id: Literal["Qwen/Qwen-Image-2.1-Turbo", "google/embeddinggemma-2"]
    ds4: NativeCapabilities


class NativeSuiteClient:
    """No residency operations: each method waits for restored service state."""

    def __init__(self, client: OpenAI) -> None:
        self._transport = client

    def models(self) -> list[NativeModel]:
        data = self._transport.get("/models", cast_to=dict[str, Any])["data"]
        return [
            NativeModel.model_validate(item)
            for item in data
            if item["id"] in (NATIVE_SUITE.image_model, NATIVE_SUITE.embedding_model)
        ]

    def upload(
        self, path: Path, *, kind: Literal["image", "video", "audio"]
    ) -> MediaContent:
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        mime = {"audio/x-wav": "audio/wav", "audio/x-flac": "audio/flac"}.get(
            mime, mime
        )
        with path.open("rb") as source:
            response = self._transport.files.create(
                file=(path.name, source, mime),
                purpose="ds4_media",  # type: ignore[arg-type]
            )
        return MediaContent(type=kind, file_id=response.id)

    def upload_image(self, image: Image.Image) -> MediaContent:
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        response = self._transport.files.create(
            file=("image.png", stream.getvalue(), "image/png"),
            purpose="ds4_media",  # type: ignore[arg-type]
        )
        return MediaContent(type="image", file_id=response.id)

    def content(self, file_id: str) -> bytes:
        return self._transport.files.content(file_id).read()

    def embeddings(
        self,
        items: Sequence[EmbeddingItem] | Sequence[str],
        *,
        dimensions: int = NATIVE_SUITE.dimensions,
        task: Literal[
            "raw", "search_query", "document", "code_query", "similarity"
        ] = "raw",
        title: str | None = None,
    ) -> dict[str, Any]:
        extension: dict[str, Any] = {"version": 1, "task": task, "overflow": "error"}
        if title is not None:
            extension["title"] = title
        response = self._transport.post(
            "/embeddings",
            cast_to=dict[str, Any],
            body={
                "model": NATIVE_SUITE.embedding_model,
                "dimensions": dimensions,
                "encoding_format": "float",
                "input": [
                    item.model_dump(exclude_none=True)
                    if isinstance(item, EmbeddingItem)
                    else item
                    for item in items
                ],
                "ds4": extension,
            },
        )
        if len(response["data"]) != len(items) or any(
            row["index"] != i or len(row["embedding"]) != dimensions
            for i, row in enumerate(response["data"])
        ):
            raise ValueError(NATIVE_SUITE.invalid_response)
        return response

    def images(
        self,
        prompt: str,
        *,
        images: Sequence[MediaContent] = (),
        n: int = 1,
        width: int = 1024,
        height: int = 1024,
        output_format: Literal["png"] = "png",
        background: Literal["auto", "opaque", "transparent"] = "auto",
        seed: int | None = None,
        idempotency_key: str | None = None,
    ) -> ImageBatch:
        extension: dict[str, Any] = {"version": 1, "sampling": "checkpoint"}
        if seed is not None:
            extension["seed"] = seed
        body: dict[str, Any] = {
            "model": NATIVE_SUITE.image_model,
            "prompt": prompt,
            "n": n,
            "size": f"{width}x{height}",
            "output_format": output_format,
            "background": background,
            "ds4": extension,
        }
        if images:
            body["images"] = [item.file_id for item in images]
        response = self._transport.post(
            "/images/edits" if images else "/images/generations",
            cast_to=dict[str, Any],
            body=body,
            options={
                "headers": {"Idempotency-Key": idempotency_key or str(uuid.uuid4())}
            },
        )
        return ImageBatch.model_validate(response)


_suite_factory: ContextVar[Callable[[], NativeSuiteClient] | None] = ContextVar(
    "native_suite_factory", default=None
)


@contextmanager
def native_suite_context(factory: Callable[[], NativeSuiteClient]) -> Iterator[None]:
    token = _suite_factory.set(factory)
    try:
        yield
    finally:
        _suite_factory.reset(token)


@contextmanager
def native_suite_client() -> Iterator[NativeSuiteClient]:
    """Borrow the calling provider's transport, or own one for this operation."""
    factory = _suite_factory.get()
    if factory is not None:
        yield factory()
    else:
        settings = DEFAULT_PROVIDER_SETTINGS
        with OpenAI(
            base_url=os.environ[settings.base_url_env_var],
            api_key=os.getenv(settings.api_key_env_var) or settings.fallback_api_key,
            timeout=NATIVE_SUITE.timeout_seconds,
        ) as transport:
            yield NativeSuiteClient(transport)
