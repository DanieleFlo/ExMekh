"""Render bounded PDF page ranges into independent images using PDFium."""

from __future__ import annotations

from contextlib import closing
from math import ceil
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pypdfium2 as pdfium

from exmekh_lib.config.file_inspection import (
    FILE_INSPECTION_TEXT,
    FileInspectionSettings,
)
from exmekh_lib.core.models.tool import AttachedToolResult
from exmekh_lib.core.models.user_message import (
    ImagesUserMessage,
    UserMessage,
    UserMessagePart,
)

from .errors import FileInspectionError
from .workspace import FileSnapshot

# PDFium forbids concurrent calls even for unrelated documents in different nodes.
_PDFIUM_LOCK = Lock()


def validate_pdf_read_arguments(
    start_page: int, end_page: int, settings: FileInspectionSettings
) -> None:
    for name, value in (("start_index", start_page), ("end_index", end_page)):
        if type(value) is not int or value < 1:
            raise FileInspectionError(
                FILE_INSPECTION_TEXT.positive_page_index.format(name=name)
            )
    if start_page > end_page:
        raise FileInspectionError(FILE_INSPECTION_TEXT.ordered_page_indices)
    for name, value in (
        ("pdf_render_dpi", settings.pdf_render_dpi),
        ("max_pdf_render_pages", settings.max_pdf_render_pages),
    ):
        if type(value) is not int or value <= 0:
            raise FileInspectionError(
                FILE_INSPECTION_TEXT.positive_setting.format(name=name)
            )
    if end_page - start_page + 1 > settings.max_pdf_render_pages:
        raise FileInspectionError(
            FILE_INSPECTION_TEXT.page_limit.format(
                max_pdf_render_pages=settings.max_pdf_render_pages
            )
        )


def read_pdf(
    snapshot: FileSnapshot,
    start_page: int,
    end_page: int,
    settings: FileInspectionSettings,
) -> AttachedToolResult:
    validate_pdf_read_arguments(start_page, end_page, settings)
    if snapshot.content is None:
        raise FileInspectionError(FILE_INSPECTION_TEXT.pdf_byte_limit)
    # Validation and prerequisite checks must not need the native renderer.
    import pypdfium2 as pdfium

    try:
        with _PDFIUM_LOCK, pdfium.PdfDocument(snapshot.content) as document:
            page_count = len(document)
            if page_count == 0:
                raise FileInspectionError(FILE_INSPECTION_TEXT.empty_pdf)
            if end_page > page_count:
                raise FileInspectionError(
                    FILE_INSPECTION_TEXT.page_range_outside_pdf.format(
                        page_count=page_count
                    )
                )
            pages, parts = _render_pages(
                document, start_page, end_page, snapshot.relative_path, settings
            )
    except FileInspectionError:
        raise
    except (pdfium.PdfiumError, OSError, ValueError, OverflowError) as error:
        raise FileInspectionError(
            FILE_INSPECTION_TEXT.pdf_render_failed.format(
                error_type=type(error).__name__, error=error
            )
        ) from error
    return AttachedToolResult(
        {
            "ok": True,
            "path": snapshot.relative_path,
            "family": "document",
            "format": "pdf",
            "start_index": start_page,
            "end_index": end_page,
            "total_pages": page_count,
            "rendered_pages": len(pages),
            "dpi": settings.pdf_render_dpi,
            "pages": pages,
        },
        UserMessage(parts),
    )


def _render_pages(
    document: pdfium.PdfDocument,
    start_page: int,
    end_page: int,
    path: str,
    settings: FileInspectionSettings,
) -> tuple[list[dict[str, int]], list[UserMessagePart]]:
    scale = settings.pdf_render_dpi / 72
    pages: list[dict[str, int]] = []
    parts: list[UserMessagePart] = []
    total_pixels = 0
    for number in range(start_page, end_page + 1):
        with closing(document[number - 1]) as page:
            width, height = page.get_size()
            total_pixels += ceil(width * scale) * ceil(height * scale)
            if total_pixels > settings.max_image_pixels:
                raise FileInspectionError(FILE_INSPECTION_TEXT.pdf_pixel_limit)
            # PDFium supports fractional scale; its untyped default is inferred as int.
            with closing(
                page.render(scale=scale, rev_byteorder=True)  # pyright: ignore[reportArgumentType]
            ) as bitmap:
                image = bitmap.to_pil().copy()
            pages.append({"page": number, "width": image.width, "height": image.height})
            parts.extend(
                [
                    FILE_INSPECTION_TEXT.pdf_attachment_label.format(
                        path=path, page=number
                    ),
                    ImagesUserMessage([image]),
                ]
            )
    return pages, parts
