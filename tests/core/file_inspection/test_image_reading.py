"""Direct image delivery, ignored indices and recoverable image read failures."""

import base64
from dataclasses import replace
from io import BytesIO

import pytest
from PIL import Image

from exmekh_lib.config.file_inspection import FILE_INSPECTION
from exmekh_lib.core.file_inspection import image_reading, reading
from exmekh_lib.core.file_inspection.errors import FileInspectionError
from exmekh_lib.core.file_inspection.reading import read_file
from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.tool import AttachedToolResult, Tool
from exmekh_lib.core.models.tool_hooks import ToolHooks
from exmekh_lib.core.providers.to_openai_chat_completion_request_mapper import (
    ToOpenAIChatCompletionRequestMapper,
)
from exmekh_lib.core.providers.to_openai_response_request_mapper import (
    ToOpenAIResponseRequestMapper,
)
from exmekh_lib.interfaces.tool_hook import AfterTool, BeforeTool
from exmekh_lib.utils.xml import stringify_tool_result
from tests.core.file_inspection.test_reading import (
    assert_paired,
    batch,
    call,
    make_node,
    open_files,
    read_call,
    replies,
    result_for,
)
from tests.file_inspection_support import inspect_file
from tests.tool_result_support import parse_tool_result


def image_from_message(message):
    return next(
        part for part in message.iter_content_parts() if isinstance(part, Image.Image)
    )


def attached_image(result):
    return image_from_message(result.attachments)


@pytest.mark.parametrize("extension", FILE_INSPECTION.image_extensions)
def test_configured_images_are_decoded_without_inspection(tmp_path, extension):
    path = tmp_path / f"picture{extension}"
    Image.new("RGB", (32, 32), (20, 40, 60)).save(path)
    result = read_file(tmp_path, path.name, start_index=0, end_index=0)
    assert isinstance(result, AttachedToolResult)
    assert result["ok"] and result["family"] == "image"
    assert result["width"] == result["height"] == 32
    assert result["start_index"] == result["end_index"] == 0
    image = attached_image(result)
    assert image.mode == "RGBA"
    assert image.size == (32, 32)
    path.unlink()
    image.load()  # Delivery owns decoded pixels, not an open file/stream.
    assert "attachments" not in result
    assert "Image.Image" not in stringify_tool_result(result)


@pytest.mark.parametrize("start,end", [(0, 0), (1, 500), (-5, -10), (20, 1)])
def test_image_indices_are_ignored_even_when_invalid_for_text(tmp_path, start, end):
    path = tmp_path / "picture.PNG"
    Image.new("RGBA", (2, 3), (10, 20, 30, 40)).save(path)
    result = read_file(tmp_path, path.name, start, end)
    assert result["start_index"] == result["end_index"] == 0
    assert attached_image(result).getpixel((0, 0)) == (10, 20, 30, 40)


def test_inspection_and_reading_share_configured_image_extensions(
    tmp_path, monkeypatch
):
    path = tmp_path / "picture.custom"
    Image.new("RGB", (2, 3)).save(path, format="PNG")
    settings = replace(FILE_INSPECTION, image_extensions=(".custom",))
    monkeypatch.setattr(reading, "FILE_INSPECTION", settings)
    report = inspect_file(tmp_path, path.name, settings=settings)
    result = read_file(tmp_path, path.name, 0, 0)
    assert report["family"] == result["family"] == "image"
    assert report["format"] == result["format"] == "png"


def test_animated_images_deliver_only_the_first_frame(tmp_path):
    path = tmp_path / "animation.gif"
    Image.new("RGB", (2, 3), "red").save(
        path, save_all=True, append_images=[Image.new("RGB", (2, 3), "blue")]
    )
    result = read_file(tmp_path, path.name, 0, 0)
    assert result["frame"] == 1
    assert attached_image(result).getpixel((0, 0)) == (255, 0, 0, 255)


def test_exif_orientation_is_applied_before_model_delivery(tmp_path):
    path = tmp_path / "rotated.jpg"
    orientation = Image.Exif()
    orientation[274] = 6
    Image.new("RGB", (2, 3), "red").save(path, exif=orientation)
    result = read_file(tmp_path, path.name, 0, 0)
    assert (result["width"], result["height"]) == attached_image(result).size == (3, 2)


@pytest.mark.parametrize(
    "limit,error",
    [("max_file_bytes", "byte limit"), ("max_image_pixels", "pixel limit")],
)
def test_image_resource_limits_are_explicit(tmp_path, monkeypatch, limit, error):
    path = tmp_path / "picture.png"
    Image.new("RGB", (2, 3)).save(path)
    monkeypatch.setattr(
        reading, "FILE_INSPECTION", replace(FILE_INSPECTION, **{limit: 1})
    )
    with pytest.raises(FileInspectionError, match=error):
        read_file(tmp_path, path.name, 0, 0)


@pytest.mark.parametrize("maximum", [1, 4])
def test_pillow_decompression_bombs_become_read_errors(tmp_path, monkeypatch, maximum):
    path = tmp_path / "picture.png"
    Image.new("RGB", (2, 3)).save(path)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", maximum)
    with pytest.raises(
        FileInspectionError, match="Image could not be read.*DecompressionBomb"
    ):
        read_file(tmp_path, path.name, 0, 0)


def test_pixel_decode_failure_is_a_read_error(tmp_path, monkeypatch):
    path = tmp_path / "picture.png"
    Image.new("RGB", (2, 3)).save(path)

    def fail(*args):
        raise OSError("truncated pixel data")

    monkeypatch.setattr(image_reading.ImageOps, "exif_transpose", fail)
    with pytest.raises(FileInspectionError, match="Image could not be read or decoded"):
        read_file(tmp_path, path.name, 0, 0)


@pytest.mark.parametrize("raw", [None, b"", b"not an image"])
def test_failed_image_reads_return_errors_without_automatic_inspection(tmp_path, raw):
    if raw is not None:
        (tmp_path / "broken.png").write_bytes(raw)
    node, provider = make_node(
        tmp_path,
        [
            open_files(),
            batch(read_call(path="broken.png", start_index=0, end_index=0)),
            AssistantMessage("done"),
        ],
    )
    assert node.run("read image").get_message() == "done"
    assert (
        result_for(provider.requests[-1], "read")["error_kind"] == "invalid_file_read"
    )
    assert all(m.get_name() != "inspect-file" for m in replies(provider.requests[-1]))
    assert replies(provider.requests[-1])[-1].get_attachments() is None
    assert_paired(provider.requests[-1].get_messages())


def test_images_respect_workroot_and_symlink_boundaries(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.png"
    Image.new("RGB", (2, 3)).save(outside)
    (tmp_path / "escape.png").symlink_to(outside)
    for path in (str(outside), "escape.png"):
        with pytest.raises(FileInspectionError, match="workroot"):
            read_file(tmp_path, path, 0, 0)


def test_image_attachments_reach_both_providers_and_cached_rereads(tmp_path):
    path = tmp_path / "picture.png"
    Image.new("RGBA", (2, 3), (10, 20, 30, 40)).save(path)
    steps = []

    class Before(BeforeTool):
        def handle(self, context):
            steps.append("before")

    class After(AfterTool):
        def handle(self, context, result):
            assert isinstance(result, AttachedToolResult)
            steps.append("after")
            return result

    node, provider = make_node(
        tmp_path,
        [
            open_files(),
            batch(read_call(path=path.name, start_index=0, end_index=0)),
            AssistantMessage("done"),
        ],
        hooks={"read-file": ToolHooks(before=Before(), after=After())},
    )
    node.run("show image")
    request = provider.requests[-1]
    response = replies(request)[-1]
    assert steps == ["before", "after"]
    assert response.get_name() == "read-file"
    assert result_for(request, "read")["family"] == "image"
    assert all(m.get_name() != "inspect-file" for m in replies(request))
    assert image_from_message(response.get_attachments()).getpixel((0, 0)) == (
        10,
        20,
        30,
        40,
    )
    for mapper, key, kind in (
        (ToOpenAIChatCompletionRequestMapper, "messages", "image_url"),
        (ToOpenAIResponseRequestMapper, "input", "input_image"),
    ):
        payload = mapper.map_request(request, model="test")
        image_part = next(
            part
            for message in payload[key]
            if isinstance(message.get("content"), list)
            for part in message["content"]
            if part.get("type") == kind
        )
        image_url = image_part["image_url"]
        url = image_url if isinstance(image_url, str) else image_url["url"]
        with Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1]))) as image:
            assert image.getpixel((0, 0)) == (10, 20, 30, 40)
    output_id = next(
        r.id for r in node._runtime._tool_cache.get_all() if r.tool_call_id == "read"
    )
    path.unlink()
    saved = node.read_tool_output(output_id)
    assert parse_tool_result(saved.content) == result_for(request, "read")
    assert attached_image(saved).getpixel((0, 0)) == (10, 20, 30, 40)
    assert_paired(request.get_messages())


def test_structured_attachments_are_also_preserved_for_managed_tools(tmp_path):
    def work(text):
        return read_file(tmp_path, "picture.png", 0, 0)

    Image.new("RGB", (2, 3), "red").save(tmp_path / "picture.png")
    node, provider = make_node(
        tmp_path,
        [batch(call("mock-tool", "image", text="read")), AssistantMessage("done")],
        overrides={"mock-tool": Tool(work, name="mock-tool")},
    )
    node.run("read image")
    response = replies(provider.requests[-1])[-1]
    assert image_from_message(response.get_attachments()).getpixel((0, 0)) == (
        255,
        0,
        0,
        255,
    )
    record = next(
        r for r in node._runtime._tool_cache.get_all() if r.tool_call_id == "image"
    )
    assert record.result["family"] == "image"
    assert attached_image(node.read_tool_output(record.id)).getpixel((0, 0)) == (
        255,
        0,
        0,
        255,
    )
