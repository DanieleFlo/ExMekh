import os
from unittest.mock import MagicMock

import pytest
from openai import OpenAI
from pydantic import BaseModel

from qs.core.models.client import Client
from qs.core.models.history import MessageHistory
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.structured_provider_request import StructuredProviderRequest
from qs.core.models.structured_provider_response import StructuredProviderResponse
from qs.core.models.system_message import SystemMessage
from qs.core.models.user_message import UserMessage
from tests.history_validation import validate_request_history

_LIVE_TEST_ENV_VAR = "QS_RUN_LIVE_TESTS"
_TRUTHY_VALUES = {"1", "true", "yes", "on"}

pytest_plugins = ("pytester",)


@pytest.fixture(autouse=True)
def validate_client_request_histories(monkeypatch: pytest.MonkeyPatch) -> None:
    original_call = Client._call_provider
    original_structured_call = Client._call_structured_provider

    def call_with_valid_history(
        client: Client, request: ProviderRequest
    ) -> ProviderResponse:
        validate_request_history(request)
        return original_call(client, request)

    monkeypatch.setattr(Client, "_call_provider", call_with_valid_history)

    def call_structured_with_valid_history[ResponseModel: BaseModel](
        client: Client, request: StructuredProviderRequest[ResponseModel]
    ) -> StructuredProviderResponse[ResponseModel]:
        validate_request_history(request)
        return original_structured_call(client, request)

    monkeypatch.setattr(
        Client, "_call_structured_provider", call_structured_with_valid_history
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    live_group = parser.getgroup("live tests")
    live_group.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="run tests that contact the configured live model provider",
    )


def pytest_xdist_auto_num_workers(config: pytest.Config) -> int:
    """Use the logical CPUs available to pytest, unless explicitly overridden."""
    requested = os.getenv("PYTEST_XDIST_AUTO_NUM_WORKERS")
    return int(requested) if requested else (os.process_cpu_count() or 1)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    env_value = os.getenv(_LIVE_TEST_ENV_VAR, "").strip().lower()
    run_live = config.getoption("--run-live") or env_value in _TRUTHY_VALUES
    skip_live = pytest.mark.skip(
        reason=(f"live test disabled; pass --run-live or set {_LIVE_TEST_ENV_VAR}=1")
    )
    for item in items:
        is_live = item.get_closest_marker("live") is not None
        # xdist must see this group before it builds worker collection IDs.
        if is_live or item.get_closest_marker("serial") is not None:
            item.add_marker(pytest.mark.xdist_group("serial"))
        if is_live and not run_live:
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
