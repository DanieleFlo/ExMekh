"""Read PDF metadata without extracting text or rendering pages."""

import io

from pypdf import PdfReader
from pypdf.errors import PdfReadError
from pypdf.generic import DictionaryObject

from exmekh_lib.config.file_inspection import FILE_INSPECTION_TEXT

from .models import Analysis, RegionIndex


def analyze_pdf(raw: bytes, index: RegionIndex) -> Analysis:
    result = Analysis("pdf", "document", index)
    reader = PdfReader(io.BytesIO(raw), strict=True)
    result.details["encrypted"] = reader.is_encrypted
    if reader.is_encrypted:
        result.warnings.append(FILE_INSPECTION_TEXT.encrypted_pdf_warning)
        return result
    pages = reader.root_object["/Pages"]
    if not isinstance(pages, DictionaryObject):
        raise PdfReadError(FILE_INSPECTION_TEXT.invalid_pdf_tree)
    declared_count = pages.get("/Count")
    if not isinstance(declared_count, int) or declared_count < 0:
        raise PdfReadError(FILE_INSPECTION_TEXT.invalid_pdf_count)
    result.details["page_count"] = declared_count
    if declared_count > index.settings.max_pdf_pages:
        result.status = "skipped_limit"
        result.warnings.append(FILE_INSPECTION_TEXT.pdf_traversal_warning)
    else:
        result.details["page_count"] = len(reader.pages)
        if reader.pages:
            page = reader.pages[0]
            result.details["first_page_size_points"] = {
                "width": float(page.mediabox.width),
                "height": float(page.mediabox.height),
            }
    metadata = reader.metadata
    if metadata is not None:
        result.details["properties"] = {
            name: str(metadata[key])[: index.settings.max_label_chars]
            for name, key in (
                ("title", "/Title"),
                ("author", "/Author"),
                ("subject", "/Subject"),
                ("creator", "/Creator"),
                ("producer", "/Producer"),
                ("created", "/CreationDate"),
                ("modified", "/ModDate"),
            )
            if key in metadata
        }
    return result
