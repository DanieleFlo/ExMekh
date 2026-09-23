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
