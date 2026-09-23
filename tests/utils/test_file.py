import base64
import json
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from qs.utils.file import (
    convert_bytes_to_base64,
    read_file_from_path,
    read_json_from_path,
    write_json_atomically,
)


def test_write_json_atomically_replaces_file(tmp_path: Path) -> None:
    target = tmp_path / "data.json"
    target.write_text("original")

    write_json_atomically(target, {"value": "café"})

    assert read_json_from_path(target) == {"value": "café"}
    assert list(tmp_path.iterdir()) == [target]


def test_failed_atomic_replace_preserves_file_and_cleans_up(tmp_path: Path) -> None:
    target = tmp_path / "data.json"
    target.write_text("original")
    with patch.object(Path, "replace", side_effect=OSError), pytest.raises(OSError):
        write_json_atomically(target, {"value": 1})
    assert target.read_text() == "original"
    assert list(tmp_path.iterdir()) == [target]


def test_failed_json_serialization_preserves_file_and_cleans_up(tmp_path: Path) -> None:
    target = tmp_path / "data.json"
    target.write_text("original")
    with pytest.raises(TypeError):
        write_json_atomically(target, {"value": object()})
    assert target.read_text() == "original"
    assert list(tmp_path.iterdir()) == [target]


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
        patch.object(Path, "open", side_effect=PermissionError("denied")),
        pytest.raises(OSError, match="denied"),
    ):
        read_file_from_path(file_path)


def test_read_json_from_path_success(tmp_path: Path) -> None:
    file_path = tmp_path / "data.json"
    file_path.write_text('{"key": "value", "numbers": [1, 2, 3]}', encoding="utf-8")
    assert read_json_from_path(file_path) == {"key": "value", "numbers": [1, 2, 3]}


def test_read_json_from_path_rejects_invalid_json(tmp_path: Path) -> None:
    file_path = tmp_path / "invalid.json"
    file_path.write_text("not json content!", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_json_from_path(file_path)


def test_read_json_from_path_delegates_to_read_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_json_from_path(tmp_path / "missing.json")
