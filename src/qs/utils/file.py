import base64
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


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
        raise TypeError("Input must be of type 'bytes'")
    base64_data = base64.b64encode(file_bytes).decode("utf-8")
    return base64_data


def read_file_from_path(file_path: Path) -> str:
    if not isinstance(file_path, Path):
        raise TypeError("File path must be a pathlib.Path object")
    if not file_path.exists():
        raise FileNotFoundError(f"File path {file_path} does not exist.")
    if not file_path.is_file():
        raise ValueError(f"File path {file_path} is not a file.")

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as error:
        raise OSError(f"An error occurred while reading the file: {error}") from error


def read_json_from_path(file_path: Path) -> Any:
    """Read and deserialize JSON content from a file path."""
    content = read_file_from_path(file_path)
    return json.loads(content)
