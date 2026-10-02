from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from uuid import uuid4

from qs.config.runtime import RUNTIME
from qs.core.cache.node_cache import NodeCache
from qs.core.cache.tool_call_record import CalledToolMetadata, ToolCallRecord
from qs.core.cache.user_message_codec import serialize_user_message
from qs.core.models.user_message import UserMessage
from qs.core.tool_calls.execution_history import ExecutionHistory
from qs.core.tool_calls.execution_manager import ToolExecutionManager
from qs.utils.json import to_json_value

from .types import CacheIdentity, ToolExecutionResult


class ToolResultRecorder:
    """Keep invocation history and cached outputs synchronized with tool work."""

    def __init__(
        self,
        history: ExecutionHistory,
        cache_provider: Callable[[], NodeCache],
        execution_manager: ToolExecutionManager,
    ) -> None:
        self._history = history
        self._cache_provider = cache_provider
        self._execution_manager = execution_manager

    def record_call(
        self, call: CalledToolMetadata, thinking: str | None
    ) -> CacheIdentity:
        entry = self._history.record(call.name, call.call_id)
        return CacheIdentity(
            name=call.name,
            source="Tool",
            tool_call_id=call.call_id,
            args=to_json_value(call.arguments),
            history_position=entry.position,
            thinking=thinking,
        )

    def persist_result(
        self,
        result: ToolExecutionResult,
        identity: CacheIdentity,
        skill_path: str,
    ) -> None:
        if result.execution is not None and result.output_id is None:
            raise RuntimeError("Managed tool execution is missing its output ID")
        record = None
        if result.save_result:
            identity = result.cache_identity or identity
            attachments = result.response.get_attachments()
            record = ToolCallRecord(
                name=identity.name,
                id=result.output_id or uuid4().hex,
                tool_call_id=identity.tool_call_id,
                args=identity.args,
                result=result.raw_result,
                skill_path=skill_path,
                thinking=identity.thinking,
                source=identity.source,
                used_tools=result.used_tools,
                user_message=(
                    serialize_user_message(attachments)
                    if attachments is not None
                    else None
                ),
                execution=(
                    result.execution.status_info()
                    if result.execution is not None
                    else None
                ),
            )
            cache = self._cache_provider()
            cache.store(record)
            self._history.attach_output_at_position(
                identity.history_position, record.id, cache.session_dir
            )
        if result.execution is not None:
            assert result.output_id is not None
            self._execution_manager.attach_output(
                result.execution.call_id, result.output_id, record
            )

    def persist_steering(
        self, message: UserMessage, output_id: str, skill_path: str
    ) -> None:
        record = ToolCallRecord(
            name=RUNTIME.steering_output_name,
            id=output_id,
            args={},
            result=message.message_dump()["content"],
            skill_path=skill_path,
            source="UserMessage",
            user_message=serialize_user_message(message),
        )
        cache = self._cache_provider()
        cache.store(record)
        entry = self._history.record(record.name, output_id)
        entry.attach_output(record.id, cache.session_dir)

    def sync_completed(self) -> None:
        for snapshot, record in self._execution_manager.completed_to_persist():
            attachments = snapshot.output_attachments()
            self._cache_provider().store(
                replace(
                    record,
                    result=snapshot.record_result(),
                    execution=snapshot.status_info(),
                    user_message=(
                        serialize_user_message(attachments)
                        if attachments is not None
                        else None
                    ),
                )
            )
            self._execution_manager.mark_persisted(record.id)
