import os
from unittest.mock import MagicMock

import pytest
from openai import OpenAI

from qs.core.models.history import MessageHistory
from qs.core.models.system_message import SystemMessage
from qs.core.models.user_message import UserMessage

_LIVE_TEST_ENV_VAR = "QS_RUN_LIVE_TESTS"
_TRUTHY_VALUES = {"1", "true", "yes", "on"}


def pytest_addoption(parser: pytest.Parser) -> None:
    live_group = parser.getgroup("live tests")
    live_group.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="run tests that contact the configured live model provider",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    env_value = os.getenv(_LIVE_TEST_ENV_VAR, "").strip().lower()
    if config.getoption("--run-live") or env_value in _TRUTHY_VALUES:
        return

    skip_live = pytest.mark.skip(
        reason=(f"live test disabled; pass --run-live or set {_LIVE_TEST_ENV_VAR}=1")
    )
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


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
    return MagicMock(spec=OpenAI)
