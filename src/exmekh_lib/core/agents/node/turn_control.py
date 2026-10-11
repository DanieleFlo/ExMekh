from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from threading import Lock
from typing import cast

from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.core.tool_calls.execution_manager import ToolExecutionManager

from .types import NodeUserMessage


class TurnControl:
    """Coordinate run admission, steering acceptance and tool wait release."""

    def __init__(self, execution_manager: ToolExecutionManager) -> None:
        self._execution_manager = execution_manager
        self._run_lock = Lock()
        self._steering_lock = Lock()
        self._accepting_steering = False
        self._pending_messages: list[UserMessage] = []

    @contextmanager
    def run(self) -> Generator[None]:
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError(RUNTIME.concurrent_run_message)
        try:
            with self._steering_lock:
                self._accepting_steering = True
            yield
        finally:
            with self._steering_lock:
                self._accepting_steering = False
                self._pending_messages.clear()
                self._execution_manager.resume_waits()
            self._run_lock.release()

    def steer(self, user_message: NodeUserMessage) -> None:
        message = cast(UserMessage, UserMessage(content=user_message).copy())
        with self._steering_lock:
            if not self._accepting_steering:
                raise RuntimeError(RUNTIME.inactive_steering_message)
            self._pending_messages.append(message)
            self._execution_manager.release_waits()

    def take_user_messages(self) -> tuple[UserMessage, ...]:
        with self._steering_lock:
            messages = tuple(self._pending_messages)
            self._pending_messages.clear()
            self._execution_manager.resume_waits()
            return messages

    def has_pending_messages(self) -> bool:
        with self._steering_lock:
            return bool(self._pending_messages)

    def try_close(self) -> bool:
        """Stop acceptance atomically only when all accepted input is consumed."""
        with self._steering_lock:
            if self._pending_messages:
                return False
            self._accepting_steering = False
            return True
