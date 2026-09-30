from unittest.mock import MagicMock

import pytest

from qs.core.agents.node.turn_control import TurnControl
from qs.core.tool_calls.execution_manager import ToolExecutionManager


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
