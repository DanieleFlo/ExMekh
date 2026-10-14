"""Decode one bounded image snapshot for model delivery and cached rereads."""

import warnings
from io import BytesIO

from PIL import Image, ImageOps

from exmekh_lib.config.file_inspection import (
    FILE_INSPECTION_TEXT,
    FileInspectionSettings,
)
from exmekh_lib.core.models.tool import AttachedToolResult
from exmekh_lib.core.models.user_message import ImagesUserMessage, UserMessage

from .errors import FileInspectionError
from .workspace import FileSnapshot


def read_image(
    snapshot: FileSnapshot, settings: FileInspectionSettings
) -> AttachedToolResult:
    if snapshot.content is None:
        raise FileInspectionError(
            FILE_INSPECTION_TEXT.image_byte_limit, code="read_limit_exceeded"
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(snapshot.content)) as source:
                if source.width * source.height > settings.max_image_pixels:
                    raise FileInspectionError(
                        FILE_INSPECTION_TEXT.image_pixel_limit,
                        code="read_limit_exceeded",
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
            FILE_INSPECTION_TEXT.image_decode_failed.format(
                error_type=type(error).__name__
            ),
            code="file_decode_failed",
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
            "start_index": 0,
            "end_index": 0,
        },
        UserMessage(
            [
                FILE_INSPECTION_TEXT.image_attachment_label.format(
                    path=snapshot.relative_path
                ),
                ImagesUserMessage([image]),
            ]
        ),
    )
