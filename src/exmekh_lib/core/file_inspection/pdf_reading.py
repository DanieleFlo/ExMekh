"""Render bounded PDF page ranges into independent images using PDFium."""

from __future__ import annotations

from contextlib import closing
from math import ceil
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pypdfium2 as pdfium

from exmekh_lib.config.file_inspection import FileInspectionSettings
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
    for name, value in (("start_line", start_page), ("end_line", end_page)):
        if type(value) is not int or value < 1:
            raise FileInspectionError(
                f"{name} must be a positive PDF page number (1-based)."
            )
    if start_page > end_page:
        raise FileInspectionError(
            "start_line must be <= end_line for the inclusive PDF page range."
        )
    for name, value in (
        ("pdf_render_dpi", settings.pdf_render_dpi),
        ("max_pdf_render_pages", settings.max_pdf_render_pages),
    ):
        if type(value) is not int or value <= 0:
            raise FileInspectionError(f"{name} must be a positive integer.")
    if end_page - start_page + 1 > settings.max_pdf_render_pages:
        raise FileInspectionError(
            f"PDF reads allow at most {settings.max_pdf_render_pages} pages per call; "
            "choose a smaller start_line/end_line interval."
        )


def read_pdf(
    snapshot: FileSnapshot,
    start_page: int,
    end_page: int,
    settings: FileInspectionSettings,
) -> AttachedToolResult:
    validate_pdf_read_arguments(start_page, end_page, settings)
    if snapshot.content is None:
        raise FileInspectionError(
            "PDF exceeds the configured file byte limit; use a smaller PDF."
        )
    # Validation and prerequisite checks must not need the native renderer.
    import pypdfium2 as pdfium

    try:
        with _PDFIUM_LOCK, pdfium.PdfDocument(snapshot.content) as document:
            page_count = len(document)
            if page_count == 0:
                raise FileInspectionError("PDF is empty; no pages can be read.")
            if end_page > page_count:
                raise FileInspectionError(
                    f"start_line/end_line are outside the PDF page range (1..{page_count})."
                )
            pages, parts = _render_pages(
                document, start_page, end_page, snapshot.relative_path, settings
            )
    except FileInspectionError:
        raise
    except (pdfium.PdfiumError, OSError, ValueError, OverflowError) as error:
        raise FileInspectionError(
            f"PDF could not be read or rendered ({type(error).__name__}: {error}); "
            "check the file, encryption and render settings, then retry."
        ) from error
    return AttachedToolResult(
        {
            "ok": True,
            "path": snapshot.relative_path,
            "family": "document",
            "format": "pdf",
            "start_page": start_page,
            "end_page": end_page,
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
                raise FileInspectionError(
                    "PDF pages exceed the configured total pixel limit; "
                    "read fewer pages or lower the configured DPI."
                )
            # PDFium supports fractional scale; its untyped default is inferred as int.
            with closing(
                page.render(scale=scale, rev_byteorder=True)  # pyright: ignore[reportArgumentType]
            ) as bitmap:
                image = bitmap.to_pil().copy()
            pages.append({"page": number, "width": image.width, "height": image.height})
            parts.extend([f"PDF {path}, page {number}", ImagesUserMessage([image])])
    return pages, parts
