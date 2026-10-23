"""Registered image generation/editing entrypoint."""

from pathlib import Path
from typing import Literal

from exmekh_lib.config.default_capabilities import DEFAULT_CAPABILITIES
from exmekh_lib.config.native_suite import NATIVE_SUITE, ImageCount, ImageDimension
from exmekh_lib.core.agents.tools.decorator import deep_tool
from exmekh_lib.core.models.native_media_tools import generate_images


@deep_tool(
    name=DEFAULT_CAPABILITIES.generate_image, description=NATIVE_SUITE.image_description
)
def run(
    work_root: Path,
    prompt: str = "",
    images: list[str] | None = None,
    n: ImageCount = 1,
    width: ImageDimension = NATIVE_SUITE.image_default_dimension,
    height: ImageDimension = NATIVE_SUITE.image_default_dimension,
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
        images=images,
        n=n,
        width=width,
        height=height,
        output_format=output_format,
        background=background,
        seed=seed,
        idempotency_key=idempotency_key,
        include_image_in_result=include_image_in_result,
        preview_scale=preview_scale,
    )
