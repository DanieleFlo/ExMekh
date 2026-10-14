from pathlib import Path
from typing import Protocol


class FileReadAccess(Protocol):
    """Authorize a file read and select its byte budget without reading content."""

    def read_limit(self, path: Path, default: int) -> int: ...
