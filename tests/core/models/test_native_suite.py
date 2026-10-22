"""Native media transport, explicit tools and attachment behavior."""

import base64
import inspect
import io
import json
from contextlib import nullcontext
from pathlib import Path

import httpx2
import pytest
from openai import OpenAI
from PIL import Image

from exmekh_lib.config.capabilities import CAPABILITY_FORMAT
from exmekh_lib.config.provider import DEFAULT_PROVIDER_SETTINGS
from exmekh_lib.core.agents.tools.deep_tools import DeepTools
from exmekh_lib.core.models import native_suite
from exmekh_lib.core.models.native_media_tools import (
    generate_images,
    workspace_media,
)
from exmekh_lib.core.models.native_suite import (
    EmbeddingItem,
    MediaContent,
    NativeSuiteClient,
    TextContent,
    native_suite_client,
    native_suite_context,
)
from exmekh_lib.core.models.tool import AttachedToolResult


def transport(handler):
    return NativeSuiteClient(
        OpenAI(
            base_url="http://native/v1",
            api_key="test",
            http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        )
    )


def test_order_task_and_no_residency_commands():
    requests = []

    def handler(request):
        requests.append(request)
        body = json.loads(request.content)
        assert body["input"][0]["content"] == [
            {"type": "text", "text": "first"},
            {"type": "video", "file_id": "file-video", "audio": "include"},
            {"type": "text", "text": "last"},
        ]
        assert body["ds4"] == {
            "version": 1,
            "task": "document",
            "overflow": "error",
            "title": "Title",
        }
        return httpx2.Response(
            200, json={"data": [{"index": 0, "embedding": [1.0] * 128}]}
        )

    suite = transport(handler)
    result = suite.embeddings(
        [
            EmbeddingItem(
                content=[
                    TextContent(text="first"),
                    MediaContent(type="video", file_id="file-video", audio="include"),
                    TextContent(text="last"),
                ]
            )
        ],
        dimensions=128,
        task="document",
        title="Title",
    )
    assert len(result["data"]) == 1
    assert [r.url.path for r in requests] == ["/v1/embeddings"]
    with native_suite_context(lambda: suite):
        with native_suite_client() as borrowed:
            assert borrowed is suite


def test_registered_tools_have_their_own_entrypoints(tmp_path):
    tools = DeepTools(
        [
            CAPABILITY_FORMAT.bundled_root / "tools" / name
            for name in ("generate-image",)
        ],
        work_root=tmp_path,
    )
    loaded = tools.load()
    assert not (CAPABILITY_FORMAT.bundled_root / "tools/embed-content/main.py").exists()
    assert len(loaded) == 1
    tools.unload()
    for method in (
        NativeSuiteClient.images,
        NativeSuiteClient.embeddings,
        generate_images,
    ):
        assert "model" not in inspect.signature(method).parameters
    assert "size" not in inspect.signature(NativeSuiteClient.images).parameters
    assert {"width", "height"} <= inspect.signature(
        NativeSuiteClient.images
    ).parameters.keys()


def test_multiple_images_preview_configuration_and_workspace_paths(tmp_path):
    stream = io.BytesIO()
    Image.new("RGBA", (32, 32), "red").save(stream, "PNG")
    encoded = base64.b64encode(stream.getvalue()).decode()
    requests = []

    def handler(request):
        requests.append(request)
        assert request.headers["Idempotency-Key"]
        body = json.loads(request.content)
        assert body["n"] == 2
        return httpx2.Response(
            200,
            json={
                "created": 1,
                "ds4": {"revision": "test", "profile": "bf16"},
                "data": [
                    {
                        "b64_json": encoded,
                        "ds4": {
                            "file_id": f"file-{i}",
                            "seed": 17 + i,
                            "width": 32,
                            "height": 32,
                            "recipe": {"steps": 8},
                        },
                    }
                    for i in range(2)
                ],
            },
        )

    suite = transport(handler)
    with native_suite_context(lambda: suite):
        result = generate_images(
            tmp_path, "red", n=2, width=32, height=32, preview_scale=0.5
        )
        assert isinstance(result, AttachedToolResult)
        parts = list(result.attachments.iter_content_parts())
        assert isinstance(parts[0], str)
        previews = [part for part in parts if isinstance(part, Image.Image)]
        assert [item.size for item in previews] == [(16, 16), (16, 16)]
        for row in result["images"]:
            assert Path(row["path"]).read_bytes() == stream.getvalue()
        no_pixels = generate_images(tmp_path, "red", n=2, include_image_in_result=False)
        assert not isinstance(no_pixels, AttachedToolResult)
    with pytest.raises(ValueError):
        workspace_media(tmp_path, "../escape.png")
    assert all(r.url.path == "/v1/images/generations" for r in requests)


def test_upload_uses_native_file_purpose_and_returns_reference(tmp_path):
    path = tmp_path / "clip.wav"
    path.write_bytes(b"RIFF-fixture")

    def handler(request):
        assert request.url.path == "/v1/files"
        assert b"ds4_media" in request.content
        assert b"RIFF-fixture" in request.content
        return httpx2.Response(
            200,
            json={
                "id": "file-audio",
                "object": "file",
                "bytes": 12,
                "created_at": 1,
                "filename": "clip.wav",
                "purpose": "ds4_media",
            },
        )

    media = transport(handler).upload(path, kind="audio")
    assert media.type == "audio" and media.file_id == "file-audio"


def test_invalid_embedding_shape_is_rejected():
    suite = transport(lambda r: httpx2.Response(200, json={"data": []}))
    with pytest.raises(ValueError):
        suite.embeddings(["text"])


@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_native_transport_lifetime_matches_its_owner(monkeypatch, borrowed, fails):
    owner = OpenAI(
        base_url="http://native/v1",
        api_key="test",
        http_client=httpx2.Client(
            transport=httpx2.MockTransport(
                lambda request: httpx2.Response(200, json={"data": []})
            )
        ),
    )
    monkeypatch.setenv(DEFAULT_PROVIDER_SETTINGS.base_url_env_var, "http://native/v1")
    monkeypatch.setattr(native_suite, "OpenAI", lambda **kwargs: owner)
    context = (
        native_suite_context(lambda: NativeSuiteClient(owner))
        if borrowed
        else nullcontext()
    )
    try:
        with context:
            with pytest.raises(RuntimeError, match="stop") if fails else nullcontext():
                with native_suite_client() as suite:
                    assert suite.models() == []
                    if fails:
                        raise RuntimeError("stop")
        assert owner.is_closed() is (not borrowed)
    finally:
        owner.close()


def test_all_reference_paths_are_checked_before_upload(tmp_path):
    path = tmp_path / "reference.png"
    path.write_bytes(b"image")
    requests = []
    suite = transport(lambda request: requests.append(request))
    with native_suite_context(lambda: suite), pytest.raises(ValueError):
        generate_images(tmp_path, "red", images=["reference.png", "../escape.png"])
    assert requests == []
