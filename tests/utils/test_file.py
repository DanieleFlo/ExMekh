import base64
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from qs.utils.file import convert_bytes_to_base64, read_file_from_path


@pytest.mark.parametrize(
    ("file_bytes", "expected"),
    [
        (b"", ""),
        (b"hello", "aGVsbG8="),
        (bytes(range(8)), "AAECAwQFBgc="),
    ],
)
def test_convert_bytes_to_base64_returns_utf8_base64(file_bytes, expected) -> None:
    result = convert_bytes_to_base64(file_bytes)

    assert result == expected
    assert base64.b64decode(result) == file_bytes


def test_convert_bytes_to_base64_rejects_non_bytes() -> None:
    with pytest.raises(TypeError, match="Input must be of type 'bytes'"):
        convert_bytes_to_base64(cast(bytes, "content"))


def test_read_file_from_path_rejects_non_path() -> None:
    with pytest.raises(TypeError, match="must be a pathlib.Path"):
        read_file_from_path(cast(Path, "file.txt"))


def test_read_file_from_path_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        read_file_from_path(tmp_path / "file.txt")


def test_read_file_from_path_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="is not a file"):
        read_file_from_path(tmp_path)


def test_read_file_from_path_wraps_read_error(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_text("content")

    with (
        patch("builtins.open", side_effect=PermissionError("denied")),
        pytest.raises(OSError, match="denied"),
    ):
        read_file_from_path(file_path)
