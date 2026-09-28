from threading import Event

import pytest

from qs.core.models.tool import ExecutionOptions, ToolExecutionOperation
from qs.core.tool_calls.execution_manager import ToolExecutionManager
from qs.core.tool_calls.output_reader import ToolOutputReader


def unavailable_cache():
    raise AssertionError("Transient output must not depend on the cache")


def test_runtime_output_is_read_without_opening_the_cache_and_acknowledges_delivery():
    release = Event()
    manager = ToolExecutionManager()
    manager.start(
        "call",
        "work",
        {},
        ToolExecutionOperation(lambda: release.wait() and "done"),
        ("owner",),
        ExecutionOptions(mode="background"),
        output_id="output",
    )
    reader = ToolOutputReader(unavailable_cache, manager)
    try:
        assert "<response-type>running</response-type>" in reader.read("output").content
    finally:
        release.set()
    assert manager.wait("call", 1).is_finished
    assert manager.next_delivery(("owner",)) is not None
    assert reader.read("output").content.endswith("done")
    assert manager.next_delivery(("owner",)) is None


def test_unsuccessful_final_read_does_not_acknowledge_delivery():
    release = Event()
    manager = ToolExecutionManager()
    manager.start(
        "call",
        "work",
        {},
        ToolExecutionOperation(release.wait),
        ("owner",),
        ExecutionOptions(mode="background"),
        output_id="output",
    )
    release.set()
    assert manager.wait("call", 1).is_finished
    reader = ToolOutputReader(unavailable_cache, manager)
    with pytest.raises(ValueError, match="does not support cursors"):
        reader.read("output", cursor=0)
    assert manager.next_delivery(("owner",)) is not None
