from qs import main


def test_main_prints_greeting(capsys) -> None:
    main()

    assert capsys.readouterr().out == "Hello from qs!\n"
