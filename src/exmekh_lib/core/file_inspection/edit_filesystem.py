"""Bounded snapshots and cooperative, atomic file commits."""

import codecs
import os
import shutil
import stat
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock, RLock
from uuid import uuid4
from weakref import WeakValueDictionary

from exmekh_lib.config.file_editing import FILE_EDITING
from exmekh_lib.config.file_inspection import FILE_INSPECTION, FILE_INSPECTION_TEXT

from .errors import FileInspectionError
from .workspace import (
    FileIdentity,
    FileSnapshot,
    read_snapshot,
    resolve_work_root,
    validate_file_path,
)

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
    root = resolve_work_root(work_root)
    candidate = root / path
    # A dangling symlink is a resource error, not authorization to create.
    try:
        candidate.lstat()
    except FileNotFoundError:
        try:
            parent = candidate.parent.resolve(strict=True)
        except FileNotFoundError as error:
            raise FileInspectionError(
                FILE_EDITING.parent_required, code="parent_directory_missing"
            ) from error
        target = parent / candidate.name
    else:
        target = candidate.resolve(strict=True)
    if not target.is_relative_to(root):
        raise FileInspectionError(
            FILE_INSPECTION_TEXT.path_outside_root, code="path_outside_workroot"
        )
    if not target.parent.is_dir():
        raise FileInspectionError(
            FILE_EDITING.parent_required, code="parent_directory_missing"
        )
    return target


def edit_snapshot(work_root: Path, path: str) -> FileSnapshot:
    snapshot = read_snapshot(
        work_root,
        path,
        replace(FILE_INSPECTION, max_file_bytes=FILE_EDITING.max_snapshot_bytes),
    )
    if snapshot.content is None:
        raise FileInspectionError(
            FILE_EDITING.snapshot_byte_limit, code="file_limit_exceeded"
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


def new_file(work_root: Path, path: str, content: str) -> dict[str, object]:
    """Publish a complete new file exclusively, without inspection evidence."""
    if (
        not isinstance(content, str)
        or len(content) < FILE_EDITING.min_new_file_characters
    ):
        raise FileInspectionError(
            FILE_EDITING.minimum_content.format(
                minimum=FILE_EDITING.min_new_file_characters
            ),
            code="invalid_content",
        )
    if "\x00" in content:
        raise FileInspectionError(FILE_EDITING.content_nul, code="invalid_content")
    encoded = content.encode("utf-8")
    if len(encoded) > FILE_EDITING.max_snapshot_bytes:
        raise FileInspectionError(
            FILE_EDITING.creation_byte_limit, code="file_limit_exceeded"
        )
    target = resolve_edit_target(work_root, path)
    with resource_lock(target):
        if os.path.lexists(work_root.resolve() / path):
            raise FileInspectionError(
                FILE_EDITING.path_exists, code="file_already_exists"
            )
        FileWriteTarget(work_root, path, target).commit(encoded)
    return {
        "ok": True,
        "path": target.relative_to(work_root.resolve()).as_posix(),
        "operation": "create",
        "size_bytes": len(encoded),
    }


def delete_file(work_root: Path, path: str) -> dict[str, object]:
    """Remove a regular file or directory tree without following nested symlinks."""
    return FileDeletionTarget.resolve(work_root, path).remove()


@dataclass(frozen=True, slots=True)
class FileDeletionTarget:
    """Own the authorized resource, identity checks and descriptor-relative removal."""

    work_root: Path
    resource: str
    path: Path

    @classmethod
    def resolve(cls, work_root: Path, resource: str) -> "FileDeletionTarget":
        validate_file_path(resource)
        if Path(resource).is_absolute():
            raise FileInspectionError(
                FILE_EDITING.deletion_relative_path, code="invalid_path"
            )
        require_descriptor_operations(os.open, os.stat, os.unlink)
        root = resolve_work_root(work_root)
        target = resolve_edit_target(root, resource)
        if target == root:
            raise FileInspectionError(
                FILE_EDITING.deletion_workroot, code="invalid_path"
            )
        if (root / resource).is_symlink():
            raise FileInspectionError(
                FILE_EDITING.deletion_target, code="invalid_deletion_target"
            )
        return cls(root, resource, target)

    def remove(self) -> dict[str, object]:
        with resource_lock(self.path):
            initial = self.path.stat()
            is_directory = stat.S_ISDIR(initial.st_mode)
            self._validate_kind(initial)
            descriptor = _open_parent_directory(self.work_root, self.path)
            try:
                self._validate_unchanged(initial, descriptor)
                if is_directory:
                    shutil.rmtree(self.path.name, dir_fd=descriptor)
                else:
                    os.unlink(self.path.name, dir_fd=descriptor)
            finally:
                os.close(descriptor)
        return {
            "ok": True,
            "path": self.path.relative_to(self.work_root).as_posix(),
            "operation": "remove",
            "size_bytes": 0,
        }

    def _validate_kind(self, initial: os.stat_result) -> None:
        if not (stat.S_ISREG(initial.st_mode) or stat.S_ISDIR(initial.st_mode)):
            raise FileInspectionError(
                FILE_EDITING.deletion_target, code="invalid_deletion_target"
            )
        if stat.S_ISDIR(initial.st_mode):
            require_descriptor_operations(os.rmdir)
            if not shutil.rmtree.avoids_symlink_attacks:
                raise FileInspectionError(
                    FILE_EDITING.unsupported_platform, code="unsupported_platform"
                )

    def _validate_unchanged(self, initial: os.stat_result, descriptor: int) -> None:
        current = os.stat(self.path.name, dir_fd=descriptor, follow_symlinks=False)
        parent = self.path.parent.stat()
        opened_parent = os.fstat(descriptor)
        if (
            resolve_edit_target(self.work_root, self.resource) != self.path
            or FileIdentity.from_stat(initial) != FileIdentity.from_stat(current)
            or stat.S_IFMT(initial.st_mode) != stat.S_IFMT(current.st_mode)
            or (parent.st_dev, parent.st_ino)
            != (opened_parent.st_dev, opened_parent.st_ino)
        ):
            raise FileInspectionError(
                FILE_EDITING.changed_before_deletion, code="file_changed"
            )


def require_descriptor_operations(*operations: Callable[..., object]) -> None:
    if not set(operations).issubset(os.supports_dir_fd):
        raise FileInspectionError(
            FILE_EDITING.unsupported_platform, code="unsupported_platform"
        )


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


@dataclass(frozen=True, slots=True)
class FileWriteTarget:
    """Bind a workspace resource to its canonical path and initial snapshot."""

    work_root: Path
    resource: str
    path: Path
    snapshot: FileSnapshot | None = None

    def validate(self) -> None:
        """Recheck canonical path, initial bytes, identity and platform permissions."""
        if resolve_edit_target(self.work_root, self.resource) != self.path:
            raise FileInspectionError(FILE_EDITING.changed, code="file_changed")
        if self.snapshot is None:
            try:
                self.path.lstat()
            except FileNotFoundError:
                pass
            else:
                raise FileInspectionError(FILE_EDITING.changed, code="file_changed")
        elif not self.snapshot.matches(edit_snapshot(self.work_root, self.resource)):
            raise FileInspectionError(FILE_EDITING.changed, code="file_changed")
        require_descriptor_operations(os.open, os.unlink, os.link, os.rename)
        if self.snapshot is not None and not os.access(self.path, os.W_OK):
            raise FileInspectionError(
                FILE_EDITING.file_not_writable, code="permission_denied"
            )
        if not os.access(self.path.parent, os.W_OK | os.X_OK):
            raise FileInspectionError(
                FILE_EDITING.parent_not_writable, code="permission_denied"
            )

    def commit(self, content: bytes) -> None:
        """Revalidate and publish under this target's cooperative lock."""
        with resource_lock(self.path):
            self._commit(content)

    def _commit(self, content: bytes) -> None:
        """Commit after an exact-byte and identity check. External writers must cooperate."""

        self.validate()
        parent_fd = _open_parent_directory(self.work_root, self.path)
        parent_identity = os.fstat(parent_fd)

        def check_parent() -> None:
            current = self.path.parent.stat()
            if (current.st_dev, current.st_ino) != (
                parent_identity.st_dev,
                parent_identity.st_ino,
            ):
                raise FileInspectionError(FILE_EDITING.changed, code="file_changed")

        temporary: str | None = None
        try:
            temporary = f"{FILE_EDITING.temporary_file_prefix}{uuid4().hex}"
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                FILE_EDITING.temporary_file_mode,
                dir_fd=parent_fd,
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                if self.snapshot is not None and hasattr(os, "fchmod"):
                    os.fchmod(stream.fileno(), stat.S_IMODE(self.snapshot.stat.st_mode))
                os.fsync(stream.fileno())
            self.validate()
            check_parent()
            if self.snapshot is None:
                # An exclusive hard link publishes complete bytes without overwriting.
                os.link(
                    temporary,
                    self.path.name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            else:
                os.replace(
                    temporary,
                    self.path.name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                temporary = None
        finally:
            try:
                if temporary is not None:
                    os.unlink(temporary, dir_fd=parent_fd)
            finally:
                os.close(parent_fd)
