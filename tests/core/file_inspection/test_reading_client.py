"""Complete mock client flows for text, image and PDF reads through the skill."""

import pytest

from exmekh_lib.core.models.assistant_message import AssistantMessage
from tests.core.agents.node.test_node import SequenceProvider
from tests.core.file_inspection.test_reading import batch, call, open_files
from tests.file_reading_support import (
    initialize_reading_node,
    make_read_fixture,
    verify_read_client_flow,
)
from tests.live.model_support import CacheTestTokenizer


@pytest.mark.parametrize("kind", ["text", "image", "pdf"])
def test_mock_client_reads_files_and_returns_skill_output(tmp_path, kind):
    fixture = make_read_fixture(tmp_path, kind)
    responses = [open_files()]
    if kind != "image":
        responses.append(batch(call("inspect-file", "inspect", path=fixture.path.name)))
    responses.extend(
        [
            batch(call("read-file", "read", **fixture.arguments())),
            batch(call("exit_agent", "exit", text=fixture.answer, items=[])),
            AssistantMessage(fixture.answer),
        ]
    )
    provider = SequenceProvider(
        responses, CacheTestTokenizer("mock-file-reading", tmp_path / "cache")
    )
    try:
        node = initialize_reading_node(provider, tmp_path)
        response = node.run(fixture.prompt)
        assert response.get_message() == fixture.answer
        verify_read_client_flow(provider.requests, node, fixture)
        assert node.depth == 0
        assert response.skill_executions[0].history
    finally:
        provider.get_session().close()
