from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from exmekh_lib.config.utility_text import UTILITY_TEXT


@dataclass(frozen=True, slots=True)
class FileIdentity:
    """Filesystem identity and version, independent of access time and aliases."""

    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> FileIdentity:
        return cls(
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    def matches(self, path: Path) -> bool:
        try:
            return not path.is_symlink() and self == self.from_stat(path.stat())
        except OSError:
            return False


def write_json_atomically(file_path: Path, value: object) -> None:
    """Replace a JSON file only after serialization succeeds.

    The parent directory must already exist. Temporary files are created beside
    the destination to keep replacement on the same filesystem.
    """
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=file_path.parent,
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=True)
            stream.close()
            temporary_path.replace(file_path)
        finally:
            stream.close()
            temporary_path.unlink(missing_ok=True)


def convert_bytes_to_base64(file_bytes: bytes) -> str:
    if not isinstance(file_bytes, bytes):
        raise TypeError(UTILITY_TEXT.input_must_be_of_type_bytes)
    base64_data = base64.b64encode(file_bytes).decode("utf-8")
    return base64_data


def read_file_from_path(file_path: Path) -> str:
    if not isinstance(file_path, Path):
        raise TypeError(UTILITY_TEXT.file_path_must_be_a_pathlib_path_object)
    if not file_path.exists():
        raise FileNotFoundError(
            UTILITY_TEXT.file_path_does_not_exist.format(file_path=file_path)
        )
    if not file_path.is_file():
        raise ValueError(
            UTILITY_TEXT.file_path_is_not_a_file.format(file_path=file_path)
        )

    try:
        with file_path.open(encoding="utf-8") as file:
            return file.read()
    except Exception as error:
        raise OSError(
            UTILITY_TEXT.an_error_occurred_while_reading_the_file.format(error=error)
        ) from error


def read_json_from_path(file_path: Path) -> Any:
    """Read and deserialize JSON content from a file path."""
    content = read_file_from_path(file_path)
    return json.loads(content)
