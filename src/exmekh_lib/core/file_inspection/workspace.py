"""Resolve workspace paths and read bounded snapshots of regular files."""

from __future__ import annotations

import mimetypes
import os
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from exmekh_lib.config.file_inspection import (
    FILE_INSPECTION_TEXT,
    FileInspectionSettings,
)
from exmekh_lib.core.file_inspection.output_access import output_file_read_limit
from exmekh_lib.utils.file import FileIdentity
from exmekh_lib.utils.tool_errors import error_diagnostic

from .errors import FileInspectionError


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    path: Path
    relative_path: str
    stat: os.stat_result
    content: bytes | None
    is_symlink: bool

    def matches(self, other: FileSnapshot) -> bool:
        """Compare complete bytes and file identity; partial snapshots never match."""
        return (
            self.content is not None
            and other.content is not None
            and self.path == other.path
            and self.content == other.content
            and FileIdentity.from_stat(self.stat) == FileIdentity.from_stat(other.stat)
        )

    def metadata(self) -> dict[str, object]:
        value = self.stat
        birth = getattr(value, "st_birthtime", None)
        return {
            "name": self.path.name,
            "extension": self.path.suffix.casefold(),
            "mime_type": mimetypes.guess_type(self.path.name, strict=False)[0],
            "is_symlink": self.is_symlink,
            "mode": stat.filemode(value.st_mode),
            "modified_at": datetime.fromtimestamp(value.st_mtime, UTC).isoformat(),
            "metadata_changed_at": datetime.fromtimestamp(
                value.st_ctime, UTC
            ).isoformat(),
            "created_at": datetime.fromtimestamp(birth, UTC).isoformat()
            if birth is not None
            else None,
        }


def validate_file_path(path: str) -> None:
    if not isinstance(path, str) or not path.strip():
        raise FileInspectionError(
            FILE_INSPECTION_TEXT.path_required, code="invalid_path"
        )
    if "\x00" in path:
        raise FileInspectionError(FILE_INSPECTION_TEXT.path_nul, code="invalid_path")


def resolve_work_root(work_root: Path) -> Path:
    """Validate the common filesystem boundary for inspection and writes."""
    try:
        root = work_root.resolve(strict=True)
        if not root.is_dir():
            raise FileInspectionError(
                FILE_INSPECTION_TEXT.root_not_directory, code="workroot_unavailable"
            )
    except (OSError, RuntimeError) as error:
        raise FileInspectionError(
            FILE_INSPECTION_TEXT.root_unavailable, code="workroot_unavailable"
        ) from error
    return root


def read_snapshot(
    work_root: Path, path: str, settings: FileInspectionSettings
) -> FileSnapshot:
    validate_file_path(path)
    root = resolve_work_root(work_root)
    # Whitespace can be part of a filename; only use strip for blank validation.
    candidate = root / path
    try:
        target = candidate.resolve(strict=True)
        relative = target.relative_to(root).as_posix()
    except ValueError as error:
        raise FileInspectionError(
            FILE_INSPECTION_TEXT.path_outside_root, code="path_outside_workroot"
        ) from error
    except (OSError, RuntimeError) as error:
        raise FileInspectionError(
            FILE_INSPECTION_TEXT.path_unavailable,
            code=error_diagnostic(error)["error_code"]
            if isinstance(error, OSError)
            else "path_unavailable",
        ) from error

    try:
        try:
            max_file_bytes = output_file_read_limit(target, settings.max_file_bytes)
        except ValueError as error:
            raise FileInspectionError(
                str(error), code="output_access_denied"
            ) from error
        flags = (
            os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        # Check before opening a special file, then check again on the descriptor.
        if not stat.S_ISREG(target.stat().st_mode):
            raise FileInspectionError(
                FILE_INSPECTION_TEXT.regular_file_required, code="not_a_regular_file"
            )
        with os.fdopen(os.open(target, flags), "rb") as stream:
            initial = os.fstat(stream.fileno())
            if not stat.S_ISREG(initial.st_mode):
                raise FileInspectionError(
                    FILE_INSPECTION_TEXT.regular_file_required,
                    code="not_a_regular_file",
                )
            content = (
                stream.read(max_file_bytes + 1)
                if initial.st_size <= max_file_bytes
                else None
            )
            if content is not None and len(content) > max_file_bytes:
                content = None
            if FileIdentity.from_stat(initial) != FileIdentity.from_stat(
                os.fstat(stream.fileno())
            ):
                raise FileInspectionError(
                    FILE_INSPECTION_TEXT.file_changed, code="file_changed"
                )
        if FileIdentity.from_stat(initial) != FileIdentity.from_stat(target.stat()):
            raise FileInspectionError(
                FILE_INSPECTION_TEXT.file_changed, code="file_changed"
            )
    except OSError as error:
        raise FileInspectionError(
            FILE_INSPECTION_TEXT.file_unreadable,
            code=error_diagnostic(error)["error_code"],
        ) from error
    return FileSnapshot(target, relative, initial, content, candidate.is_symlink())
