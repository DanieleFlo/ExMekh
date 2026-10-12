"""Bounded snapshots and cooperative, atomic file commits."""

import codecs
import os
import stat
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from threading import Lock, RLock
from uuid import uuid4
from weakref import WeakValueDictionary

from exmekh_lib.config.file_editing import FILE_EDITING
from exmekh_lib.config.file_inspection import FILE_INSPECTION

from .errors import FileInspectionError
from .workspace import FileSnapshot, read_snapshot, validate_file_path

_locks: WeakValueDictionary[str, RLock] = WeakValueDictionary()
_registry_lock = Lock()


@contextmanager
def resource_lock(path: Path) -> Generator[None]:
    with _registry_lock:
        lock = _locks.setdefault(str(path), RLock())
    with lock:
        yield


def resolve_edit_target(work_root: Path, path: str) -> Path:
    validate_file_path(path)
    root = work_root.resolve(strict=True)
    if not root.is_dir():
        raise FileInspectionError("workroot must be a directory.")
    candidate = root / path
    # A dangling symlink is a resource error, not authorization to create.
    try:
        candidate.lstat()
    except FileNotFoundError:
        parent = candidate.parent.resolve(strict=True)
        target = parent / candidate.name
    else:
        target = candidate.resolve(strict=True)
    if not target.is_relative_to(root):
        raise FileInspectionError("path must remain inside the workroot.")
    if not target.parent.is_dir():
        raise FileInspectionError("The parent directory must already exist.")
    return target


def edit_snapshot(work_root: Path, path: str) -> FileSnapshot:
    snapshot = read_snapshot(
        work_root,
        path,
        replace(FILE_INSPECTION, max_file_bytes=FILE_EDITING.max_snapshot_bytes),
    )
    if snapshot.content is None:
        raise FileInspectionError(
            "File exceeds the edit snapshot byte limit; no truncated snapshot is allowed."
        )
    return snapshot


def encode_replacement(text: str, original: bytes, encoding: str) -> bytes:
    for bom, codec in (
        (codecs.BOM_UTF32_LE, "utf-32-le"),
        (codecs.BOM_UTF32_BE, "utf-32-be"),
        (codecs.BOM_UTF16_LE, "utf-16-le"),
        (codecs.BOM_UTF16_BE, "utf-16-be"),
    ):
        if original.startswith(bom):
            return bom + text.encode(codec)
    return text.encode(encoding)


def _open_parent_directory(work_root: Path, target: Path) -> int:
    root = work_root.resolve(strict=True)
    relative = target.parent.relative_to(root)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(root, flags)
    try:
        for component in relative.parts:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def validate_edit_target(
    work_root: Path,
    original_path: str,
    target: Path,
    snapshot: FileSnapshot | None,
    content: bytes | None,
) -> None:
    """Recheck canonical path, initial bytes, identity and platform permissions."""
    if resolve_edit_target(work_root, original_path) != target:
        raise FileInspectionError(FILE_EDITING.changed)
    if snapshot is None:
        try:
            target.lstat()
        except FileNotFoundError:
            pass
        else:
            raise FileInspectionError(FILE_EDITING.changed)
    elif not snapshot.matches(edit_snapshot(work_root, original_path)):
        raise FileInspectionError(FILE_EDITING.changed)
    if not {os.open, os.unlink, os.link, os.rename}.issubset(os.supports_dir_fd):
        raise FileInspectionError(
            "This platform lacks descriptor-relative atomic edit operations; use a supported filesystem/runtime."
        )
    if snapshot is not None and content is not None and not os.access(target, os.W_OK):
        raise FileInspectionError(
            "File is not writable; check permissions before retrying."
        )
    if not os.access(target.parent, os.W_OK | os.X_OK):
        raise FileInspectionError(
            "Parent directory is not writable/searchable; check permissions before retrying."
        )


def commit_edit(
    work_root: Path,
    original_path: str,
    target: Path,
    snapshot: FileSnapshot | None,
    content: bytes | None,
) -> None:
    """Commit after an exact-byte and identity check. External writers must cooperate."""

    def check_current() -> None:
        validate_edit_target(work_root, original_path, target, snapshot, content)

    check_current()
    parent_fd = _open_parent_directory(work_root, target)
    parent_identity = os.fstat(parent_fd)

    def check_parent() -> None:
        current = target.parent.stat()
        if (current.st_dev, current.st_ino) != (
            parent_identity.st_dev,
            parent_identity.st_ino,
        ):
            raise FileInspectionError(FILE_EDITING.changed)

    temporary: str | None = None
    try:
        if content is None:
            check_current()
            check_parent()
            os.unlink(target.name, dir_fd=parent_fd)
            return
        temporary = f".exmekh-edit-{uuid4().hex}"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            if snapshot is not None and hasattr(os, "fchmod"):
                os.fchmod(stream.fileno(), stat.S_IMODE(snapshot.stat.st_mode))
            os.fsync(stream.fileno())
        check_current()
        check_parent()
        if snapshot is None:
            # An exclusive hard link publishes complete bytes without overwriting.
            os.link(
                temporary,
                target.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        else:
            os.replace(
                temporary, target.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd
            )
            temporary = None
    finally:
        try:
            if temporary is not None:
                os.unlink(temporary, dir_fd=parent_fd)
        finally:
            os.close(parent_fd)
