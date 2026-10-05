"""Orchestrate bounded inspection and select a page of navigation regions."""

import csv
import tarfile
import zipfile
from dataclasses import asdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from pypdf.errors import PdfReadError

from exmekh_lib.config.file_inspection import FILE_INSPECTION, FileInspectionSettings

from .containers import analyze_image, analyze_tar, analyze_zip
from .models import Analysis, RegionIndex, RegionPage
from .pdf import analyze_pdf
from .selection import validate_selection
from .text import analyze_text
from .workspace import FileSnapshot, read_snapshot

_IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".avif", ".ico"}
)
_ARCHIVE_EXTENSIONS = frozenset({".tar", ".tgz", ".gz", ".bz2", ".xz"})


def inspect_file(
    work_root: Path,
    path: str,
    *,
    detailed: bool = False,
    max_items: int = FILE_INSPECTION.default_max_items,
    offset: int = 0,
    query: str | None = None,
    settings: FileInspectionSettings = FILE_INSPECTION,
) -> dict[str, object]:
    validate_selection(detailed, max_items, offset, query, settings)
    snapshot = read_snapshot(work_root, path, settings)
    analysis = _analyze(snapshot, settings)
    page = analysis.index.select(max_items=max_items, offset=offset, query=query)
    return _build_report(snapshot, analysis, page, detailed)


def _build_report(
    snapshot: FileSnapshot, analysis: Analysis, page: RegionPage, detailed: bool
) -> dict[str, object]:
    result: dict[str, object] = {
        "ok": True,
        "path": snapshot.relative_path,
        "format": analysis.format,
        "family": analysis.family,
        "size_bytes": snapshot.stat.st_size,
        "analysis_status": analysis.status,
        "analysis": analysis.details,
        **page.to_dict(),
        "warnings": analysis.warnings,
    }
    if "total_lines" in analysis.details:
        result["total_lines"] = analysis.details["total_lines"]
    if detailed:
        result["metadata"] = snapshot.metadata()
        result["limits"] = asdict(analysis.index.settings)
    return result


def _analyze(snapshot: FileSnapshot, settings: FileInspectionSettings) -> Analysis:
    index = RegionIndex(settings)
    extension = snapshot.path.suffix.casefold()
    raw = snapshot.content
    if raw is None:
        return Analysis(
            extension.lstrip(".") or "unknown",
            "unknown",
            index,
            status="skipped_limit",
            warnings=[
                "File exceeds the configured byte limit; only metadata is available."
            ],
        )
    try:
        if raw.startswith(b"PK\x03\x04") or extension in {
            ".zip",
            ".docx",
            ".xlsx",
            ".xlsm",
            ".pptx",
        }:
            return analyze_zip(raw, index)
        if extension in _ARCHIVE_EXTENSIONS or raw[257:262] == b"ustar":
            return analyze_tar(raw, index)
        if extension in _IMAGE_EXTENSIONS:
            return analyze_image(raw, index)
        if raw.startswith(b"%PDF-") or extension == ".pdf":
            return analyze_pdf(raw, index)
        return analyze_text(raw, extension, index)
    except (
        ValueError,
        SyntaxError,
        UnicodeError,
        csv.Error,
        OSError,
        KeyError,
        EOFError,
        tarfile.TarError,
        zipfile.BadZipFile,
        PdfReadError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        return Analysis(
            extension.lstrip(".") or "unknown",
            "unknown",
            index,
            status="invalid_format",
            warnings=[f"Format analysis failed: {type(error).__name__}."],
        )
