from pathlib import Path

import pytest


@pytest.fixture
def execution_suite(pytester: pytest.Pytester, request: pytest.FixtureRequest) -> Path:
    project_root = request.config.rootpath
    pytester.makeini(
        f"""
        [pytest]
        pythonpath = "{project_root.as_posix()}" "{(project_root / "src").as_posix()}"
        markers =
            live: synthetic live test without external calls
            serial: test requiring sequential execution
        """
    )
    pytester.makeconftest(
        "from tests.conftest import pytest_addoption, pytest_collection_modifyitems"
    )
    pytester.makepyfile(
        """
        from pathlib import Path

        import pytest

        def record_worker(request):
            worker = request.config.workerinput['workerid']
            Path(f'{request.node.name}.txt').write_text(worker)

        @pytest.mark.live
        def test_live(request):
            record_worker(request)

        @pytest.mark.serial
        @pytest.mark.parametrize('case', range(3))
        def test_serial(request, case):
            record_worker(request)

        @pytest.mark.parametrize('case', range(6))
        def test_parallel(request, case):
            record_worker(request)
        """
    )
    return pytester.path


@pytest.mark.parametrize("opt_in", ["disabled", "flag", "environment"])
def test_live_and_serial_tests_share_one_worker(
    pytester: pytest.Pytester,
    execution_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
    opt_in: str,
) -> None:
    monkeypatch.setenv("QS_RUN_LIVE_TESTS", "1" if opt_in == "environment" else "0")
    arguments = ["-n", "2", "--dist=loadgroup", "-q"]
    if opt_in == "flag":
        arguments.append("--run-live")

    result = pytester.runpytest_subprocess(*arguments)

    enabled = opt_in != "disabled"
    result.assert_outcomes(passed=10 if enabled else 9, skipped=0 if enabled else 1)
    serial_workers = {
        path.read_text() for path in execution_suite.glob("test_serial*.txt")
    }
    assert len(serial_workers) == 1
    live_record = execution_suite / "test_live.txt"
    if enabled:
        assert live_record.read_text() in serial_workers
    else:
        assert not live_record.exists()
    parallel_workers = {
        path.read_text() for path in execution_suite.glob("test_parallel*.txt")
    }
    assert parallel_workers == {"gw0", "gw1"}
