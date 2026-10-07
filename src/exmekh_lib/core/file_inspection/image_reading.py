"""Decode one bounded image snapshot for model delivery and cached rereads."""

import warnings
from io import BytesIO

from PIL import Image, ImageOps

from exmekh_lib.config.file_inspection import FileInspectionSettings
from exmekh_lib.core.models.tool import AttachedToolResult
from exmekh_lib.core.models.user_message import ImagesUserMessage, UserMessage

from .errors import FileInspectionError
from .workspace import FileSnapshot


def read_image(
    snapshot: FileSnapshot, settings: FileInspectionSettings
) -> AttachedToolResult:
    if snapshot.content is None:
        raise FileInspectionError(
            "Image exceeds the configured file byte limit; use a smaller image."
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(snapshot.content)) as source:
                if source.width * source.height > settings.max_image_pixels:
                    raise FileInspectionError(
                        "Image dimensions exceed the configured pixel limit; "
                        "use a smaller image."
                    )
                image_format = (source.format or "image").casefold()
                # Decode while the snapshot stream is open; deliver only frame 1.
                image = ImageOps.exif_transpose(source).convert("RGBA")
    except FileInspectionError:
        raise
    except (
        OSError,
        ValueError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise FileInspectionError(
            f"Image could not be read or decoded ({type(error).__name__}); "
            "check the file and its format, then retry."
        ) from error
    return AttachedToolResult(
        {
            "ok": True,
            "path": snapshot.relative_path,
            "family": "image",
            "format": image_format,
            "width": image.width,
            "height": image.height,
            "frame": 1,
            "start_line": 0,
            "end_line": 0,
        },
        UserMessage(
            [f"Image read from {snapshot.relative_path}", ImagesUserMessage([image])]
        ),
    )
