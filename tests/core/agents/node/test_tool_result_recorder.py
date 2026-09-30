from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from qs.core.agents.node.tool_result_recorder import ToolResultRecorder
from qs.core.agents.node.types import CacheIdentity, ToolExecutionResult
from qs.core.cache.node_cache import NodeCache
from qs.core.models.tool_message import ToolMessage
from qs.core.tool_calls.execution_history import ExecutionHistory
from qs.core.tool_calls.execution_manager import ExecutionSnapshot, ToolExecutionManager


def test_managed_result_without_output_id_fails_before_persistence():
    cache = MagicMock(spec=NodeCache)
    manager = MagicMock(spec=ToolExecutionManager)
    recorder = ToolResultRecorder(ExecutionHistory(), lambda: cache, manager)
    result = ToolExecutionResult(
        response=ToolMessage("done", "call", "work"),
        save_result=True,
        execution=ExecutionSnapshot(
            call_id="call",
            name="work",
            arguments={},
            status="completed",
            started_at=datetime.now(timezone.utc),
            finished_at=None,
            output_id=None,
            elapsed_seconds=0,
        ),
    )
    identity = CacheIdentity("work", "Tool", "call", {}, 1)

    with pytest.raises(RuntimeError, match="missing its output ID"):
        recorder.persist_result(result, identity, "/")

    cache.store.assert_not_called()
    manager.attach_output.assert_not_called()
