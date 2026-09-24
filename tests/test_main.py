from unittest.mock import Mock

from qs import main


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
    monkeypatch.setenv("QS_MODEL", "configured-model")
    monkeypatch.setenv("QS_API", "chat_completions")
    monkeypatch.setenv("QS_ROOT_HDS", "main")
    monkeypatch.setenv("QS_CHILD_HDS", "mock-sub-agent")
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
