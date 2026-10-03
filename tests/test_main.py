from unittest.mock import Mock

from exmekh_lib import main
from exmekh_lib.core.agents.deep_agent import (
    DeepAgentExecutionEvent,
    DeepAgentExecutionEventType,
    DeepAgentStreamEvent,
)
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)


def test_stream_printer_groups_tokens_by_section(capsys) -> None:
    printer = main.StreamPrinter()

    for event_type, content in (
        (ProviderStreamEventType.START, None),
        (ProviderStreamEventType.RESPONSE, "first"),
        (ProviderStreamEventType.RESPONSE, " second"),
        (ProviderStreamEventType.REASONING, "thought"),
        (ProviderStreamEventType.END, None),
        (ProviderStreamEventType.RESPONSE, "next"),
    ):
        printer.handle(
            DeepAgentStreamEvent(
                ProviderStreamEvent(event_type, content), depth=0, skill_path="/"
            )
        )

    assert capsys.readouterr().out == (
        "\nResponse: first second\nThinking: thought\n\nResponse: next"
    )


def test_stream_printer_starts_a_new_section_after_tool_execution(capsys) -> None:
    printer = main.StreamPrinter()
    response = DeepAgentStreamEvent(
        ProviderStreamEvent(ProviderStreamEventType.RESPONSE, "text"),
        depth=0,
        skill_path="/",
    )
    execution = DeepAgentExecutionEvent(
        type=DeepAgentExecutionEventType.TOOL_START,
        depth=0,
        skill_path="/",
        history=[],
        history_tree=[],
        tool_name="lookup",
        tool_call_id="call-1",
    )

    printer.handle(response)
    printer.handle(execution)
    printer.handle(response)

    assert capsys.readouterr().out == (
        "\nResponse: text\ntool_start: lookup (depth=0, path=/)\n\nResponse: text"
    )


def test_cli_registers_mock_sub_agent(monkeypatch) -> None:
    provider = Mock()
    agent = Mock()
    monkeypatch.setattr(main, "LocalProvider", Mock(return_value=provider))
    create_agent = Mock(return_value=agent)
    monkeypatch.setattr(main, "DeepAgent", create_agent)
    monkeypatch.setattr("builtins.input", lambda _: "exit")

    main.main()

    root = create_agent.call_args.kwargs["node"]
    assert root.name_hds == "main"
    assert [child.name_hds for child in root.children] == ["mock-sub-agent"]


def test_cli_reads_provider_settings(monkeypatch) -> None:
    monkeypatch.setenv("EXMEKH_LIB_MODEL", "configured-model")
    monkeypatch.setenv("EXMEKH_LIB_API", "chat_completions")
    monkeypatch.setenv("EXMEKH_LIB_ROOT_HDS", "main")
    monkeypatch.setenv("EXMEKH_LIB_CHILD_HDS", "mock-sub-agent")
    provider = Mock()
    make_provider = Mock(return_value=provider)
    monkeypatch.setattr(main, "LocalProvider", make_provider)
    create_agent = Mock(return_value=Mock())
    monkeypatch.setattr(main, "DeepAgent", create_agent)
    monkeypatch.setattr("builtins.input", lambda _: "exit")

    main.main()

    make_provider.assert_called_once_with(
        model="configured-model", api="chat_completions"
    )
    root = create_agent.call_args.kwargs["node"]
    assert root.name_hds == "main"
    assert [child.name_hds for child in root.children] == ["mock-sub-agent"]
