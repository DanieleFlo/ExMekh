"""Own temporary output files and protect live sessions across processes."""

from __future__ import annotations

import errno
import os
import shutil
import sys
from pathlib import Path
from tempfile import mkdtemp
from threading import Lock
from typing import BinaryIO, cast
from uuid import UUID, uuid4

from exmekh_lib.config.cache import CACHE_FORMAT
from exmekh_lib.config.cache_text import CACHE_TEXT
from exmekh_lib.config.tool_output import TOOL_OUTPUT_TEXT
from exmekh_lib.core.cache.paths import cache_component
from exmekh_lib.utils.file import FileIdentity

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


class _FileLease:
    def __init__(self, path: Path) -> None:
        if path.is_symlink():
            raise ValueError(CACHE_TEXT.output_lease_must_not_be_a_symlink)
        descriptor = os.open(
            path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            CACHE_FORMAT.output_lease_permissions,
        )
        self._stream = cast(
            BinaryIO, os.fdopen(descriptor, CACHE_FORMAT.output_lease_mode)
        )
        if not self._stream.read(CACHE_FORMAT.output_lease_length):
            self._stream.write(CACHE_FORMAT.output_lease_seed)
            self._stream.flush()

    def acquire(self, *, blocking: bool) -> bool:
        self._stream.seek(0)
        try:
            if sys.platform == "win32":
                msvcrt.locking(
                    self._stream.fileno(),
                    msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK,
                    CACHE_FORMAT.output_lease_length,
                )
            else:
                fcntl.flock(
                    self._stream.fileno(),
                    fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB),
                )
        except OSError as error:
            if not blocking and error.errno in (errno.EACCES, errno.EAGAIN):
                return False
            raise
        return True

    def close(self) -> None:
        self._stream.close()


class OutputFileSession:
    """One lease per cache session/workroot, with node-scoped immutable files."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        root = self.root
        self._lock = Lock()
        self._closed = False
        self._files: dict[Path, FileIdentity] = {}
        root.mkdir(parents=True, exist_ok=True)
        cleanup = _FileLease(root / CACHE_FORMAT.output_cleanup_lock)
        try:
            cleanup.acquire(blocking=True)
            self._remove_abandoned(root)
            self.directory = root / f"{CACHE_FORMAT.session_prefix}{uuid4().hex}"
            self.directory.mkdir()
            self._lease = _FileLease(
                self.directory / CACHE_FORMAT.output_lease_filename
            )
            self._lease.acquire(blocking=True)
        finally:
            cleanup.close()

    @classmethod
    def clean_abandoned(cls, root: Path) -> None:
        """Run on startup without creating storage or touching active sessions."""
        if not root.exists():
            return
        if root.is_symlink():
            raise ValueError(CACHE_TEXT.output_root_must_not_be_a_symlink)
        cleanup = _FileLease(root / CACHE_FORMAT.output_cleanup_lock)
        try:
            cleanup.acquire(blocking=True)
            cls._remove_abandoned(root)
        finally:
            cleanup.close()

    @staticmethod
    def _remove_abandoned(root: Path) -> None:
        for directory in root.glob(f"{CACHE_FORMAT.session_prefix}*"):
            suffix = directory.name.removeprefix(CACHE_FORMAT.session_prefix)
            try:
                if UUID(hex=suffix).hex != suffix:
                    continue
            except ValueError:
                continue
            path = directory / CACHE_FORMAT.output_lease_filename
            if (
                directory.is_symlink()
                or not directory.is_dir()
                or path.is_symlink()
                or not path.is_file()
            ):
                continue
            lease = _FileLease(path)
            try:
                abandoned = lease.acquire(blocking=False)
            finally:
                lease.close()
            if abandoned:
                # The cleanup lease prevents another process claiming this directory.
                shutil.rmtree(directory)

    def write(self, node_id: str, key: str, parts: tuple[str, ...]) -> tuple[Path, ...]:
        with self._lock:
            if self._closed:
                raise RuntimeError(CACHE_TEXT.output_file_session_is_closed)
            node_directory = self.directory / cache_component(node_id)
            destination = node_directory / cache_component(key)
            if destination.resolve() != destination:
                raise ValueError(
                    CACHE_TEXT.output_directory_must_not_redirect_to_another_location
                )
            node_directory.mkdir(exist_ok=True)
            paths = tuple(
                destination
                / f"{index:0{CACHE_FORMAT.output_part_digits}d}{CACHE_FORMAT.output_text_suffix}"
                for index in range(1, len(parts) + 1)
            )
            if destination.exists():
                if all(
                    path in self._files and self._files[path].matches(path)
                    for path in paths
                ):
                    return paths
                raise OSError(
                    CACHE_TEXT.previously_exported_output_files_are_unavailable
                )
            staging = Path(
                mkdtemp(prefix=CACHE_FORMAT.output_pending_prefix, dir=node_directory)
            )
            try:
                for path, text in zip(paths, parts, strict=True):
                    (staging / path.name).write_bytes(
                        text.encode(
                            CACHE_FORMAT.output_text_encoding,
                            errors=CACHE_FORMAT.output_text_errors,
                        )
                    )
                staging.replace(destination)
                for path in paths:
                    self._files[path] = FileIdentity.from_stat(path.stat())
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
            return paths

    def read_limit(self, node_id: str, path: Path, default: int) -> int:
        """Allow only registered files of this node; large multiline files are readable."""
        if not path.is_relative_to(self.root):
            return default
        with self._lock:
            node_directory = self.directory / cache_component(node_id)
            if (
                self._closed
                or not path.is_relative_to(node_directory)
                or path not in self._files
                or not self._files[path].matches(path)
            ):
                raise ValueError(TOOL_OUTPUT_TEXT.access_denied)
            return max(default, self._files[path].size)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            cleanup = _FileLease(self.root / CACHE_FORMAT.output_cleanup_lock)
            try:
                cleanup.acquire(blocking=True)
                self._lease.close()
                shutil.rmtree(self.directory)
                self._files.clear()
                self._closed = True
            finally:
                cleanup.close()
