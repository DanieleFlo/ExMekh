"""Resolve workspace paths and read bounded snapshots of regular files."""

import mimetypes
import os
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from exmekh_lib.config.file_inspection import FileInspectionSettings

from .errors import FileInspectionError


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    path: Path
    relative_path: str
    stat: os.stat_result
    content: bytes | None
    is_symlink: bool

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
            "path must contain non-space text inside the workroot."
        )
    if "\x00" in path:
        raise FileInspectionError("path must not contain NUL characters.")


def read_snapshot(
    work_root: Path, path: str, settings: FileInspectionSettings
) -> FileSnapshot:
    validate_file_path(path)
    try:
        root = work_root.resolve(strict=True)
        if not root.is_dir():
            raise FileInspectionError("workroot must identify a directory.")
    except (OSError, RuntimeError) as error:
        raise FileInspectionError(
            "workroot does not exist or cannot be resolved; restore it before retrying."
        ) from error
    # Whitespace can be part of a filename; only use strip for blank validation.
    candidate = root / path
    try:
        target = candidate.resolve(strict=True)
        relative = target.relative_to(root).as_posix()
    except ValueError as error:
        raise FileInspectionError("path must remain inside the workroot.") from error
    except (OSError, RuntimeError) as error:
        raise FileInspectionError(
            "path does not exist or cannot be resolved."
        ) from error

    try:
        flags = (
            os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        # Check before opening a special file, then check again on the descriptor.
        if not stat.S_ISREG(target.stat().st_mode):
            raise FileInspectionError("path must identify a regular file.")
        with os.fdopen(os.open(target, flags), "rb") as stream:
            initial = os.fstat(stream.fileno())
            if not stat.S_ISREG(initial.st_mode):
                raise FileInspectionError("path must identify a regular file.")
            content = (
                stream.read(settings.max_file_bytes + 1)
                if initial.st_size <= settings.max_file_bytes
                else None
            )
            if content is not None and len(content) > settings.max_file_bytes:
                content = None
            if _identity(initial) != _identity(os.fstat(stream.fileno())):
                raise FileInspectionError("File changed during inspection; retry.")
        if _identity(initial) != _identity(target.stat()):
            raise FileInspectionError("File changed during inspection; retry.")
    except OSError as error:
        raise FileInspectionError(
            "File could not be read; check permissions and retry."
        ) from error
    return FileSnapshot(target, relative, initial, content, candidate.is_symlink())


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
