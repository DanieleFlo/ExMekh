from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

from qs.utils.file import write_json_atomically


class TokenizerSession:
    """Reserve sequential indices and own their temporary storage."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._temporary_directory = TemporaryDirectory(prefix="session-", dir=directory)
        self._counter_path = Path(self._temporary_directory.name) / "counter.json"
        self._next_index = 0
        self._closed = False
        self._lock = Lock()

    def reserve_next_index(self, capacity: int) -> int:
        with self._lock:
            if self._closed:
                raise RuntimeError("Tokenizer session is closed")
            if self._next_index >= capacity:
                raise RuntimeError("Tokenizer ID space exhausted for this session")
            index = self._next_index
            # Reserve before I/O: a failed write can skip an ID, never repeat one.
            self._next_index += 1
            write_json_atomically(self._counter_path, {"next_index": self._next_index})
            return index

    def close(self) -> None:
        """Remove temporary storage and prevent further allocation."""
        with self._lock:
            self._temporary_directory.cleanup()
            self._closed = True
