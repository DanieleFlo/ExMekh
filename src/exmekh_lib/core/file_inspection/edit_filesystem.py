"""Bounded snapshots and cooperative, atomic file commits."""

import codecs
import os
import stat
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from threading import Lock, RLock
from uuid import uuid4
from weakref import WeakValueDictionary

from exmekh_lib.config.file_editing import FILE_EDITING
from exmekh_lib.config.file_inspection import FILE_INSPECTION, FILE_INSPECTION_TEXT

from .errors import FileInspectionError
from .workspace import FileIdentity, FileSnapshot, read_snapshot, validate_file_path

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
        raise FileInspectionError(FILE_INSPECTION_TEXT.root_not_directory)
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
        raise FileInspectionError(FILE_INSPECTION_TEXT.path_outside_root)
    if not target.parent.is_dir():
        raise FileInspectionError(FILE_EDITING.parent_required)
    return target


def edit_snapshot(work_root: Path, path: str) -> FileSnapshot:
    snapshot = read_snapshot(
        work_root,
        path,
        replace(FILE_INSPECTION, max_file_bytes=FILE_EDITING.max_snapshot_bytes),
    )
    if snapshot.content is None:
        raise FileInspectionError(FILE_EDITING.snapshot_byte_limit)
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


def new_file(work_root: Path, path: str, content: str) -> dict[str, object]:
    """Publish a complete new file exclusively, without inspection evidence."""
    if (
        not isinstance(content, str)
        or len(content) < FILE_EDITING.min_new_file_characters
    ):
        raise FileInspectionError(
            FILE_EDITING.minimum_content.format(
                minimum=FILE_EDITING.min_new_file_characters
            )
        )
    if "\x00" in content:
        raise FileInspectionError(FILE_EDITING.content_nul)
    encoded = content.encode("utf-8")
    if len(encoded) > FILE_EDITING.max_snapshot_bytes:
        raise FileInspectionError(FILE_EDITING.creation_byte_limit)
    target = resolve_edit_target(work_root, path)
    with resource_lock(target):
        if os.path.lexists(work_root.resolve() / path):
            raise FileInspectionError(FILE_EDITING.path_exists)
        commit_edit(work_root, path, target, None, encoded)
    return {
        "ok": True,
        "path": target.relative_to(work_root.resolve()).as_posix(),
        "operation": "create",
        "size_bytes": len(encoded),
    }


def delete_file(work_root: Path, path: str) -> dict[str, object]:
    """Remove an existing regular file using metadata, never text decoding."""
    require_descriptor_operations(os.open, os.stat, os.unlink)
    target = resolve_edit_target(work_root, path)
    with resource_lock(target):
        initial = target.stat()
        if not stat.S_ISREG(initial.st_mode):
            raise FileInspectionError(FILE_EDITING.deletion_regular_file)
        descriptor = _open_parent_directory(work_root, target)
        try:
            current = os.stat(target.name, dir_fd=descriptor, follow_symlinks=False)
            parent = target.parent.stat()
            opened_parent = os.fstat(descriptor)
            if (
                resolve_edit_target(work_root, path) != target
                or FileIdentity.from_stat(initial) != FileIdentity.from_stat(current)
                or not stat.S_ISREG(current.st_mode)
                or (parent.st_dev, parent.st_ino)
                != (opened_parent.st_dev, opened_parent.st_ino)
            ):
                raise FileInspectionError(FILE_EDITING.changed_before_deletion)
            os.unlink(target.name, dir_fd=descriptor)
        finally:
            os.close(descriptor)
    return {
        "ok": True,
        "path": target.relative_to(work_root.resolve()).as_posix(),
        "operation": "remove",
        "size_bytes": 0,
    }


def require_descriptor_operations(*operations: Callable[..., object]) -> None:
    if not set(operations).issubset(os.supports_dir_fd):
        raise FileInspectionError(FILE_EDITING.unsupported_platform)


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
    require_descriptor_operations(os.open, os.unlink, os.link, os.rename)
    if snapshot is not None and not os.access(target, os.W_OK):
        raise FileInspectionError(FILE_EDITING.file_not_writable)
    if not os.access(target.parent, os.W_OK | os.X_OK):
        raise FileInspectionError(FILE_EDITING.parent_not_writable)


def commit_edit(
    work_root: Path,
    original_path: str,
    target: Path,
    snapshot: FileSnapshot | None,
    content: bytes,
) -> None:
    """Commit after an exact-byte and identity check. External writers must cooperate."""

    def check_current() -> None:
        validate_edit_target(work_root, original_path, target, snapshot)

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
