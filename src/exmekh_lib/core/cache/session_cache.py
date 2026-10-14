from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Any, Self

from exmekh_lib.config.cache import CACHE_FORMAT
from exmekh_lib.config.cache_text import CACHE_TEXT
from exmekh_lib.core.cache.node_cache import NodeCache
from exmekh_lib.core.cache.output_file_store import OutputFileSession
from exmekh_lib.core.cache.paths import cache_component
from exmekh_lib.core.cache.tool_call_record import ToolCallRecord
from exmekh_lib.core.cache.tool_call_store import ToolCallStore
from exmekh_lib.utils.file import write_json_atomically


class SessionCache:
    """Owns temporary storage for a single process runtime session.

    Integrates:
      - Tokenizer counter synchronization (counter.json).
      - Node-scoped temporary tool call persistence (tool_calls/<node_id>/<call_id>.json).
      - Node-scoped session state and lifecycle cleanup.
    """

    def __init__(self, directory: Path) -> None:
        if not isinstance(directory, Path):
            raise TypeError(CACHE_TEXT.directory_must_be_a_path)
        directory.mkdir(parents=True, exist_ok=True)

        self._temporary_directory = TemporaryDirectory(
            prefix=CACHE_FORMAT.session_prefix, dir=directory
        )
        self._counter_path = (
            Path(self._temporary_directory.name) / CACHE_FORMAT.counter_filename
        )
        self._next_index = 0
        self._closed = False
        self._lock = Lock()

        self._session_root = Path(self._temporary_directory.name)
        self._tool_calls_dir = self._session_root / CACHE_FORMAT.tool_calls_folder
        self._node_views: dict[str, NodeCache] = {}
        self._node_states: dict[str, dict[str, Any]] = {}
        self._output_files: dict[Path, OutputFileSession] = {}

    @property
    def session_dir(self) -> Path:
        return self._session_root

    @property
    def is_closed(self) -> bool:
        return self._closed

    def reserve_next_index(self, capacity: int) -> int:
        """Reserve sequential indices and synchronize them atomically to counter.json."""
        with self._lock:
            self._ensure_active()
            self._validate_capacity(capacity)
            index = self._next_index
            # A failed write may already have replaced the file: never reuse an ID.
            self._next_index = index + 1
            self._persist_counter(self._counter_path, {"next_index": self._next_index})
            return index

    def node(self, node_id: str) -> NodeCache:
        """Get or create an isolated cache view for a specific node in this session."""
        clean_id = self._node_key(node_id)

        with self._lock:
            self._ensure_active()
            if clean_id not in self._node_views:
                self._node_views[clean_id] = NodeCache(clean_id, self)
                self._node_states.setdefault(clean_id, {})
            return self._node_views[clean_id]

    def store_tool_call(self, node_id: str, record: ToolCallRecord) -> Path:
        with self._lock:
            self._ensure_active()
            store = self._tool_store(node_id)
            return store.save(record)

    def get_tool_call(self, node_id: str, call_id: str) -> ToolCallRecord | None:
        with self._lock:
            self._ensure_active()
            store = self._tool_store(node_id)
            return store.get(call_id)

    def get_node_tool_calls(self, node_id: str) -> list[ToolCallRecord]:
        with self._lock:
            self._ensure_active()
            store = self._tool_store(node_id)
            return store.get_all()

    def list_tool_call_ids(self, node_id: str) -> list[str]:
        with self._lock:
            self._ensure_active()
            store = self._tool_store(node_id)
            return store.list_ids()

    def set_node_state(self, node_id: str, key: str, value: Any) -> None:
        """Store an arbitrary in-memory state value for a specific node in this session."""
        with self._lock:
            self._ensure_active()
            node_id = self._node_key(node_id)
            if node_id not in self._node_states:
                self._node_states[node_id] = {}
            self._node_states[node_id][key] = value

    def get_node_state(self, node_id: str, key: str, default: Any = None) -> Any:
        """Retrieve an in-memory state value for a specific node in this session."""
        with self._lock:
            self._ensure_active()
            return self._node_states.get(self._node_key(node_id), {}).get(key, default)

    def has_node_state(self, node_id: str, key: str) -> bool:
        """Check if an in-memory state key exists for a specific node in this session."""
        with self._lock:
            self._ensure_active()
            return key in self._node_states.get(self._node_key(node_id), {})

    def delete_node_state(self, node_id: str, key: str) -> bool:
        """Remove an in-memory state entry for a specific node in this session."""
        with self._lock:
            self._ensure_active()
            store = self._node_states.get(self._node_key(node_id), {})
            if key not in store:
                return False
            del store[key]
            return True

    def close(self) -> None:
        """Remove temporary storage and invalidate session (counter + all node tool calls)."""
        with self._lock:
            if self._closed:
                return
            for output_files in self._output_files.values():
                output_files.close()
            self._output_files.clear()
            self._temporary_directory.cleanup()
            self._node_states.clear()
            self._node_views.clear()
            self._closed = True

    def output_files(self, root: Path) -> OutputFileSession:
        """Own the output-file lifecycle even when tool-result caching is disabled."""
        root = root.resolve()
        with self._lock:
            self._ensure_active()
            if root not in self._output_files:
                self._output_files[root] = OutputFileSession(root)
            return self._output_files[root]

    def __enter__(self) -> Self:
        self._ensure_active()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()

    def _ensure_active(self) -> None:
        if self._closed:
            raise RuntimeError(CACHE_TEXT.session_is_closed)

    def _validate_capacity(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError(CACHE_TEXT.capacity_must_be_an_integer)
        if capacity <= 0:
            raise ValueError(CACHE_TEXT.capacity_must_be_positive)
        if self._next_index >= capacity:
            raise RuntimeError(CACHE_TEXT.tokenizer_id_space_exhausted_for_this_session)

    def _persist_counter(self, path: Path, payload: dict[str, Any]) -> None:
        write_json_atomically(path, payload)

    def _tool_store(self, node_id: str) -> ToolCallStore:
        component = cache_component(self._node_key(node_id))
        directory = self._tool_calls_dir / component
        expected_directory = (
            self._session_root.resolve() / CACHE_FORMAT.tool_calls_folder / component
        )
        if directory.resolve() != expected_directory:
            raise ValueError(
                CACHE_TEXT.node_cache_directory_must_not_redirect_to_another_location
            )
        return ToolCallStore(directory)

    @staticmethod
    def _node_key(node_id: str) -> str:
        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError(CACHE_TEXT.node_id_must_be_a_non_empty_string)
        return node_id.strip()
