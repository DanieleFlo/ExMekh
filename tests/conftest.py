from unittest.mock import MagicMock

import pytest

from qs.core.models.history import MessageHistory
from qs.core.models.system_message import SystemMessage
from qs.core.models.user_message import UserMessage


@pytest.fixture
def system_message() -> SystemMessage:
    return SystemMessage("You are helpful")


@pytest.fixture
def user_message() -> UserMessage:
    return UserMessage("Hello")


@pytest.fixture
def valid_history(system_message: SystemMessage) -> MessageHistory:
    return MessageHistory([system_message])


@pytest.fixture
def openai_client() -> MagicMock:
    return MagicMock()
