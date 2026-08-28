import base64
from pathlib import Path


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
