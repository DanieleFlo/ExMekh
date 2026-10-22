"""Registered image generation/editing entrypoint."""

from pathlib import Path
from typing import Literal

from exmekh_lib.config.default_capabilities import DEFAULT_CAPABILITIES
from exmekh_lib.config.native_suite import NATIVE_SUITE
from exmekh_lib.core.agents.tools.decorator import deep_tool
from exmekh_lib.core.models.native_media_tools import generate_images


@deep_tool(
    name=DEFAULT_CAPABILITIES.generate_image, description=NATIVE_SUITE.image_description
)
def run(
    work_root: Path,
    prompt: str = "",
    images: list[str] | None = None,
    n: int = 1,
    width: int = 1024,
    height: int = 1024,
    output_format: Literal["png"] = "png",
    background: Literal["auto", "opaque", "transparent"] = "auto",
    seed: int | None = None,
    idempotency_key: str | None = None,
    include_image_in_result: bool = NATIVE_SUITE.include_image_in_result,
    preview_scale: float = NATIVE_SUITE.preview_scale,
) -> dict[str, object]:
    return generate_images(
        work_root,
        prompt,
        images,
        n,
        width,
        height,
        output_format,
        background,
        seed,
        idempotency_key,
        include_image_in_result,
        preview_scale,
    )
