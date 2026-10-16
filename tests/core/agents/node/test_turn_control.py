from unittest.mock import MagicMock

import pytest

from exmekh_lib.core.agents.node.turn_control import TurnControl
from exmekh_lib.core.tool_calls.execution_manager import ToolExecutionManager


def test_run_scope_keeps_admission_exclusive_until_exit():
    control = TurnControl(ToolExecutionManager())
    with pytest.raises(RuntimeError, match="requires an active"):
        control.steer("before")

    with control.run():
        with pytest.raises(RuntimeError, match="already active"):
            with control.run():
                pytest.fail("A rejected run must not enter its body")
        control.steer("accepted")
        assert control.try_close() is False
        assert control.take_user_messages()[0].get_message() == "accepted"
        assert control.try_close() is True
        with pytest.raises(RuntimeError, match="requires an active"):
            control.steer("after closure")
        with pytest.raises(RuntimeError, match="already active"):
            with control.run():
                pytest.fail("Closing acceptance must not release run admission")

    with pytest.raises(RuntimeError, match="requires an active"):
        control.steer("after return")
    with control.run():
        assert control.take_user_messages() == ()


def test_queued_guidance_prevents_closure_until_delivered():
    manager = MagicMock(spec=ToolExecutionManager)
    control = TurnControl(manager)
    with control.run():
        control.steer("new correction")
        assert control.try_close() is False
        manager.release_waits.assert_called_once_with()
        manager.resume_waits.assert_not_called()

        messages = control.take_user_messages()
        assert [message.get_message() for message in messages] == [
            "new correction",
        ]
        manager.resume_waits.assert_called_once_with()
        assert control.try_close() is True


def test_failed_run_discards_pending_input_and_restores_waits():
    manager = MagicMock(spec=ToolExecutionManager)
    control = TurnControl(manager)
    with pytest.raises(OSError, match="run failed"):
        with control.run():
            control.steer("pending")
            raise OSError("run failed")

    manager.resume_waits.assert_called_once_with()
    with pytest.raises(RuntimeError, match="requires an active"):
        control.steer("inactive")
    with control.run():
        assert control.take_user_messages() == ()
        assert control.try_close() is True


def test_invalid_input_does_not_release_waits_or_queue_guidance():
    manager = MagicMock(spec=ToolExecutionManager)
    control = TurnControl(manager)
    with control.run():
        with pytest.raises(TypeError, match="User content"):
            control.steer(42)
        manager.release_waits.assert_not_called()
        assert control.try_close() is True


def test_compaction_requests_are_coalesced_and_keep_the_run_open():
    manager = MagicMock(spec=ToolExecutionManager)
    control = TurnControl(manager)
    operation = MagicMock()
    with control.run():
        control.compact_history(operation)
        control.compact_history(operation)
        operation.assert_not_called()
        assert control.try_close() is False
        control.steer("guidance")
        assert control.take_compaction_request() is True
        manager.resume_waits.assert_not_called()
        assert control.take_compaction_request() is False
        control.take_user_messages()
        assert control.try_close() is True
        with pytest.raises(RuntimeError, match="closing"):
            control.compact_history(operation)
    control.compact_history(operation)
    operation.assert_called_once_with()


def test_failed_run_discards_compaction_and_idle_compaction_has_exclusive_admission():
    control = TurnControl(ToolExecutionManager())
    with pytest.raises(OSError):
        with control.run():
            control.compact_history(lambda: None)
            raise OSError("failed")
    assert control.take_compaction_request() is False

    def compact():
        with pytest.raises(RuntimeError, match="already active"):
            with control.run():
                pytest.fail("A run must not enter during compaction")
        raise OSError("compaction failed")

    with pytest.raises(OSError, match="compaction failed"):
        control.compact_history(compact)
    with control.run():
        assert control.try_close() is True


def test_delivering_steering_keeps_waits_released_until_pending_compaction_is_taken():
    manager = MagicMock(spec=ToolExecutionManager)
    control = TurnControl(manager)
    with control.run():
        control.compact_history(lambda: None)
        control.steer("guidance")
        assert control.take_user_messages()[0].get_message() == "guidance"
        manager.resume_waits.assert_not_called()
        assert control.take_compaction_request() is True
        manager.resume_waits.assert_called_once_with()
