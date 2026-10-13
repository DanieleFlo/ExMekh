"""Inspect archive indexes and image headers without extracting content."""

import bz2
import gzip
import io
import lzma
import tarfile
import warnings
import zipfile
from pathlib import PurePosixPath, PureWindowsPath

from PIL import Image

from exmekh_lib.config.file_inspection import FILE_INSPECTION_TEXT

from .models import Analysis, RegionIndex


def analyze_zip(raw: bytes, index: RegionIndex) -> Analysis:
    result = Analysis("zip", "archive", index)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        expanded = sum(member.file_size for member in members)
        result.details.update(
            entries=len(members),
            expanded_bytes=expanded,
            encrypted_entries=sum(bool(member.flag_bits & 1) for member in members),
            unsafe_paths=sum(unsafe_member_name(member.filename) for member in members),
        )
        for member in members[: index.settings.max_archive_entries]:
            index.add("directory" if member.is_dir() else "member", member.filename)
        if len(members) > index.settings.max_archive_entries:
            index.mark_truncated()
        if expanded > index.settings.max_expanded_bytes:
            result.warnings.append(FILE_INSPECTION_TEXT.archive_declared_size_warning)
    return result


def analyze_tar(raw: bytes, index: RegionIndex) -> Analysis:
    result = Analysis("tar", "archive", index)
    entries = expanded = unsafe = 0

    content = _tar_content(raw, index.settings.max_expanded_bytes)
    if content is None:
        result.status = "skipped_limit"
        index.mark_truncated()
        result.warnings.append(FILE_INSPECTION_TEXT.archive_decompression_warning)
        return result
    with tarfile.open(fileobj=io.BytesIO(content), mode="r|") as archive:
        for member in archive:
            if entries >= index.settings.max_archive_entries:
                _mark_partial_archive(result)
                break
            entries += 1
            expanded += member.size
            unsafe += unsafe_member_name(member.name)
            index.add("directory" if member.isdir() else "member", member.name)
            if expanded > index.settings.max_expanded_bytes:
                _mark_partial_archive(result)
                break
    result.details.update(
        entries_scanned=entries,
        expanded_bytes_scanned=expanded,
        unsafe_paths_scanned=unsafe,
    )
    return result


def _mark_partial_archive(result: Analysis) -> None:
    result.index.mark_truncated()
    result.status = "skipped_limit"
    result.warnings.append(FILE_INSPECTION_TEXT.archive_partial_warning)


def unsafe_member_name(name: str) -> bool:
    normalized = name.replace("\\", "/")
    return (
        PurePosixPath(normalized).is_absolute()
        or bool(PureWindowsPath(name).drive)
        or ".." in PurePosixPath(normalized).parts
    )


def analyze_image(raw: bytes, index: RegionIndex) -> Analysis:
    result = Analysis("image", "image", index)
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(raw)) as image:
            result.format = (image.format or "image").casefold()
            result.details.update(
                width=image.width, height=image.height, mode=image.mode
            )
            # Header-only inspection: no pixel decoding, frame scan, EXIF, or OCR.
            if image.width * image.height > index.settings.max_image_pixels:
                result.status = "skipped_limit"
                result.warnings.append(FILE_INSPECTION_TEXT.image_dimensions_warning)
            else:
                result.details["animated"] = bool(getattr(image, "is_animated", False))
    return result


def _tar_content(raw: bytes, maximum: int) -> bytes | None:
    stream = io.BytesIO(raw)
    if raw.startswith(b"\x1f\x8b"):
        decompressor = gzip.GzipFile(fileobj=stream)
    elif raw.startswith(b"BZh"):
        decompressor = bz2.BZ2File(stream)
    elif raw.startswith(b"\xfd7zXZ"):
        decompressor = lzma.LZMAFile(stream)
    else:
        return raw
    try:
        with decompressor:
            content = decompressor.read(maximum + 1)
    except lzma.LZMAError as error:
        raise ValueError(FILE_INSPECTION_TEXT.invalid_xz) from error
    return content if len(content) <= maximum else None
