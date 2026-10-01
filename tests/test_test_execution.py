from pathlib import Path

import pytest

from tests.conftest import pytest_xdist_auto_num_workers


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
        from time import monotonic, sleep

        import pytest

        def record_worker(request):
            worker = request.config.workerinput['workerid']
            Path(f'{request.node.name}.txt').write_text(worker)
            return worker

        @pytest.mark.live
        def test_live(request):
            record_worker(request)

        @pytest.mark.serial
        @pytest.mark.parametrize('case', range(3))
        def test_serial(request, case):
            record_worker(request)

        @pytest.mark.parametrize('case', range(6))
        def test_parallel(request, case):
            worker = record_worker(request)
            Path(f'{worker}.parallel').touch()
            deadline = monotonic() + 10
            while len(list(Path('.').glob('*.parallel'))) < 2:
                assert monotonic() < deadline, 'Both parallel workers must start'
                sleep(0.01)
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


@pytest.mark.parametrize("cpu_count", [None, 1, 8, 32])
def test_auto_workers_use_available_logical_cpus(monkeypatch, cpu_count):
    monkeypatch.delenv("PYTEST_XDIST_AUTO_NUM_WORKERS", raising=False)
    monkeypatch.setattr("os.process_cpu_count", lambda: cpu_count)
    assert pytest_xdist_auto_num_workers(None) == (cpu_count or 1)


def test_auto_worker_override_takes_precedence(monkeypatch):
    monkeypatch.setenv("PYTEST_XDIST_AUTO_NUM_WORKERS", "16")
    monkeypatch.setattr("os.process_cpu_count", lambda: 8)
    assert pytest_xdist_auto_num_workers(None) == 16
