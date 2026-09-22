from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from qs.core.cache.tool_call_record import CalledToolMetadata, ToolCallRecord
from qs.core.models.assistant_message import ToolCall

if TYPE_CHECKING:
    from qs.core.cache.session_cache import SessionCache


class NodeCache:
    """Scoped cache view for a specific node within a SessionCache."""

    def __init__(self, node_id: str, session: SessionCache) -> None:
        self._node_id = node_id
        self._session = session

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def is_active(self) -> bool:
        return not self._session.is_closed

    @property
    def session_dir(self) -> Path:
        return self._session.session_dir

    def store(self, record: ToolCallRecord) -> Path:
        return self._session.store_tool_call(self._node_id, record)

    def record(
        self,
        name: str,
        id: str,
        args: dict[str, Any],
        result: Any,
        *,
        skill_path: str = "/",
        thinking: str | None = None,
        source: Literal["Tool", "DeepSkill"] = "Tool",
    ) -> ToolCallRecord:
        record = ToolCallRecord(
            name=name,
            id=id,
            args=args,
            result=result,
            skill_path=skill_path,
            thinking=thinking,
            source=source,
        )
        self.store(record)
        return record

    def record_call(
        self,
        tool_call: CalledToolMetadata | ToolCall,
        result: Any,
        id: str | None = None,
        *,
        skill_path: str = "/",
        thinking: str | None = None,
        source: Literal["Tool", "DeepSkill"] = "Tool",
    ) -> ToolCallRecord:
        record = ToolCallRecord.from_call(
            tool_call,
            result=result,
            id=id,
            skill_path=skill_path,
            thinking=thinking,
            source=source,
        )
        self.store(record)
        return record

    def get(self, call_id: str) -> ToolCallRecord | None:
        return self._session.get_tool_call(self._node_id, call_id)

    def get_all(self) -> list[ToolCallRecord]:
        return self._session.get_node_tool_calls(self._node_id)

    def list_ids(self) -> list[str]:
        return self._session.list_tool_call_ids(self._node_id)

    def has(self, call_id: str) -> bool:
        return self.get(call_id) is not None

    def set_state(self, key: str, value: Any) -> None:
        """Store an in-memory state value for this node."""
        self._session.set_node_state(self._node_id, key, value)

    def get_state(self, key: str, default: Any = None) -> Any:
        """Retrieve an in-memory state value for this node."""
        return self._session.get_node_state(self._node_id, key, default)

    def has_state(self, key: str) -> bool:
        """Check if an in-memory state key exists for this node."""
        return self._session.has_node_state(self._node_id, key)

    def delete_state(self, key: str) -> bool:
        """Remove an in-memory state entry for this node."""
        return self._session.delete_node_state(self._node_id, key)
