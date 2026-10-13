"""Shared files and assertions for mock and live read-file client flows."""

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw

from exmekh_lib.core.agents.node import DefaultCapabilities, Node
from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.tool_message import ToolMessage
from tests.tool_result_support import parse_tool_result


@dataclass(frozen=True)
class ReadFixture:
    path: Path
    start: int
    end: int
    answer: str
    prompt: str

    def arguments(self):
        return {
            "path": self.path.name,
            "start_index": self.start,
            "end_index": self.end,
        }


def make_read_fixture(directory: Path, kind: str) -> ReadFixture:
    if kind == "text":
        path = directory / "notes.txt"
        answer = f"READ-{uuid4().hex}"
        path.write_text(f"Ignore this line\n{answer}\nIgnore this line too\n")
        start, end = 2, 2
        instruction = (
            "Return only the exact token on line 2, without quotes or explanation."
        )
    elif kind == "image":
        path = directory / "shape.png"
        image = Image.new("RGB", (360, 360), "purple")
        ImageDraw.Draw(image).polygon([(180, 45), (45, 315), (315, 315)], fill="lime")
        image.save(path)
        start, end = 0, 0
        answer = "green|triangle|purple"
        instruction = (
            "Identify the shape's color, shape name and background color in the image. "
            "Reply only with lowercase color|shape|background, separated by |."
        )
    else:
        assert kind == "pdf"
        path = directory / "shapes.pdf"
        pages = []
        for color, shape in (
            ("yellow", "square"),
            ("red", "circle"),
            ("blue", "square"),
            ("green", "circle"),
        ):
            image = Image.new("RGB", (360, 360), "white")
            draw = ImageDraw.Draw(image)
            if shape == "circle":
                draw.ellipse((50, 50, 310, 310), fill=color)
            else:
                draw.rectangle((50, 50, 310, 310), fill=color)
            pages.append(image)
        pages[0].save(
            path, format="PDF", save_all=True, append_images=pages[1:], resolution=72
        )
        start, end = 2, 3
        answer = "red circle|blue square"
        instruction = (
            "Identify the dominant shape's color and shape name on each requested PDF page, "
            "in page order. Reply only with lowercase color shape|color shape."
        )
    inspection = (
        "Do not inspect this image; read it directly."
        if kind == "image"
        else "First inspect the file and use its line/page count before reading."
    )
    return ReadFixture(
        path,
        start,
        end,
        answer,
        f"Enter file-processing and read {path.name!r} with read-file, "
        f"start_index={start}, end_index={end}. {inspection} "
        f"{instruction} Call exit_agent with text set to that exact answer and items=[]. "
        "Return the answer itself, not output IDs or references. After the skill returns, "
        "repeat the answer as your final response and do not reopen the skill. "
        "Use the actual file results, not guesses.",
    )


def initialize_reading_node(provider, directory: Path) -> Node:
    hds = directory / "capabilities" / "hds"
    hds.mkdir(parents=True)
    (hds / "main.md").write_text(
        "---\ndescription: Read workspace files.\nexit-tool: exit_agent\n"
        "sub-skills:\n  - file-processing\n---\n"
        "Use {{!file-processing}} for requested file reads. Follow the requested output format. "
        "After the skill returns its answer, repeat that answer and stop; do not call more tools."
    )
    node = Node(
        [hds.parent],
        name_hds="main",
        persistent_history=False,
        agent_context_id=f"file-reading-{uuid4().hex}",
        default_capabilities=DefaultCapabilities(
            harness=False,
            include={
                "file-processing",
                "inspect-file",
                "read-file",
                "edit-file",
                "new-file",
                "delete-file",
                "exit_agent",
            },
        ),
    )
    node.initialize(provider, work_root=directory)
    return node


def verify_read_client_flow(requests, node, fixture: ReadFixture):
    request, message, result = _successful_read(requests)
    history = request.get_messages()
    _verify_read_call(history, message, fixture)
    assert result["path"] == fixture.path.name
    images = []
    if fixture.path.suffix == ".txt":
        assert result["lines"] == [{"line": 2, "text": fixture.answer}]
        assert message.get_attachments() is None
    else:
        images = _verify_visual_read(history, message, result, fixture)
    saved = node.read_tool_output(result_id(message.get_message()))
    assert parse_tool_result(saved.content) == result
    assert bool(saved.attachments) == bool(message.get_attachments())
    if saved.attachments is not None:
        cached_images = [
            p
            for p in saved.attachments.iter_content_parts()
            if isinstance(p, Image.Image)
        ]
        assert [image.tobytes() for image in cached_images] == [
            image.tobytes() for image in images
        ]


def _successful_read(requests):
    for request in requests:
        for message in request.get_messages():
            if isinstance(message, ToolMessage) and message.get_name() == "read-file":
                result = parse_tool_result(message.get_message())
                if result.get("ok"):
                    return request, message, result
    raise AssertionError(
        "CLIENT: no successful read-file result was delivered to the provider"
    )


def _verify_read_call(history, message, fixture: ReadFixture):
    calls = [
        c
        for m in history
        if isinstance(m, AssistantMessage)
        for c in m.get_tool_calls() or ()
    ]
    read = next(c for c in calls if c.get_id() == message.get_tool_call_id())
    assert read.get_arguments() == fixture.arguments(), (
        f"MODEL: wrong read arguments: {read.get_arguments()!r}"
    )
    if fixture.path.suffix != ".png":
        assert any(
            c.get_name() == "inspect-file"
            and c.get_arguments()["path"] == fixture.path.name
            for c in calls[: calls.index(read)]
        )
    else:
        assert all(c.get_name() != "inspect-file" for c in calls)


def _verify_visual_read(history, message, result, fixture: ReadFixture):
    companion = message.get_attachments()
    assert companion is not None and companion in history
    images = [p for p in companion.iter_content_parts() if isinstance(p, Image.Image)]
    assert len(images) == (1 if fixture.path.suffix == ".png" else 2)
    expected_colors = (
        [(0, 255, 0)] if fixture.path.suffix == ".png" else [(255, 0, 0), (0, 0, 255)]
    )
    for image, expected in zip(images, expected_colors, strict=True):
        pixel = image.getpixel((image.width // 2, image.height // 2))[:3]
        assert all(
            abs(actual - target) <= 5
            for actual, target in zip(pixel, expected, strict=True)
        )
    if fixture.path.suffix == ".pdf":
        assert [p["page"] for p in result["pages"]] == [2, 3]
        assert result["total_pages"] == 4
    return images


def result_id(content: str) -> str:
    return ET.fromstring(content.split("</header>", 1)[0] + "</header>").findtext("id")
