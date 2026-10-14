import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import psutil
import pytest

from exmekh_lib.capabilities.tools.bash import bash_process
from exmekh_lib.capabilities.tools.bash.bash_process import BashProcess
from exmekh_lib.utils.tool_errors import error_diagnostic


@pytest.mark.parametrize("discovery", [True, False])
def test_missing_bash_is_distinct_from_missing_work_file(
    tmp_path, monkeypatch, discovery
):
    monkeypatch.setattr(bash_process.shutil, "which", lambda name: None)
    process = BashProcess(
        "pwd", tmp_path, None if discovery else str(tmp_path / "absent-bash")
    )
    with pytest.raises(FileNotFoundError) as caught:
        process.run()
    assert error_diagnostic(caught.value)["error_code"] == "bash_unavailable"
    assert "Install Bash" in error_diagnostic(caught.value)["hint"]


def test_missing_bash_workroot_is_not_misreported_as_missing_shell(tmp_path):
    process = BashProcess("pwd", tmp_path / "missing", "/bin/bash")
    with pytest.raises(FileNotFoundError) as caught:
        process.run()
    assert error_diagnostic(caught.value)["error_code"] == "workroot_unavailable"


def test_cancellation_before_launch_prevents_process_creation(tmp_path, monkeypatch):
    launch = MagicMock()
    monkeypatch.setattr(bash_process.subprocess, "Popen", launch)
    process = BashProcess("printf never", tmp_path, "bash")

    assert process.cancel() is True
    with pytest.raises(RuntimeError, match="cancelled before launch"):
        process.run()
    launch.assert_not_called()


def test_missing_process_pipes_fail_explicitly(tmp_path, monkeypatch):
    child = SimpleNamespace(stdout=None, stderr=None, poll=lambda: 0)
    monkeypatch.setattr(bash_process.subprocess, "Popen", lambda *args, **kw: child)
    process = BashProcess("printf test", tmp_path, "bash")

    with pytest.raises(RuntimeError, match="did not provide stdout and stderr"):
        process.run()


def test_pipe_read_failure_is_reported_after_process_exit(tmp_path, monkeypatch):
    stdout = io.BytesIO()
    stderr = io.BytesIO()
    child = SimpleNamespace(stdout=stdout, stderr=stderr, wait=lambda: 0)
    monkeypatch.setattr(bash_process.subprocess, "Popen", lambda *args, **kw: child)
    process = BashProcess("printf test", tmp_path, "bash")

    with pytest.raises(OSError):
        process.run()
    assert stdout.closed and stderr.closed


def test_cancellation_returns_false_when_process_has_exited(tmp_path):
    process = BashProcess("printf test", tmp_path, "bash")
    process._process = SimpleNamespace(poll=lambda: 0)

    assert process.cancel() is False


@pytest.mark.parametrize("killed", [False, True])
def test_windows_cancellation_waits_only_after_confirmed_kill(
    tmp_path, monkeypatch, killed
):
    monkeypatch.setattr(bash_process, "os", SimpleNamespace(name="nt"))
    wait = MagicMock()
    process = BashProcess("printf test", tmp_path, "bash")
    process._process = SimpleNamespace(poll=lambda: None, wait=wait)
    monkeypatch.setattr(process, "_kill_windows_tree", lambda child: killed)

    assert process.cancel() is killed
    if killed:
        wait.assert_called_once_with()
    else:
        wait.assert_not_called()


@pytest.mark.parametrize("alive_after_lookup", [False, True])
def test_windows_tree_disappearance_is_handled(
    tmp_path, monkeypatch, alive_after_lookup
):
    child = SimpleNamespace(pid=123, poll=lambda: None if alive_after_lookup else 0)
    monkeypatch.setattr(
        bash_process.psutil,
        "Process",
        MagicMock(side_effect=psutil.NoSuchProcess(child.pid)),
    )

    if alive_after_lookup:
        with pytest.raises(psutil.NoSuchProcess):
            BashProcess._kill_windows_tree(child)
    else:
        assert BashProcess._kill_windows_tree(child) is False


def test_windows_tree_cancellation_tolerates_exited_descendants(monkeypatch):
    descendant = MagicMock()
    descendant.kill.side_effect = psutil.NoSuchProcess(124)
    parent = MagicMock()
    parent.children.return_value = [descendant]
    parent.kill.side_effect = psutil.NoSuchProcess(123)
    monkeypatch.setattr(bash_process.psutil, "Process", lambda pid: parent)

    assert BashProcess._kill_windows_tree(SimpleNamespace(pid=123)) is True
    parent.children.assert_called_once_with(recursive=True)


@pytest.mark.parametrize("process_exited", [False, True])
def test_unix_cancellation_handles_process_lookup_race(
    tmp_path, monkeypatch, process_exited
):
    def missing_process(pid, signal_number):
        raise ProcessLookupError

    monkeypatch.setattr(
        bash_process,
        "os",
        SimpleNamespace(name="posix", killpg=missing_process),
    )
    monkeypatch.setattr(bash_process, "signal", SimpleNamespace(SIGKILL=9))
    child = MagicMock()
    child.pid = 123
    child.poll.side_effect = [None, 0 if process_exited else None]
    process = BashProcess("sleep 1", tmp_path, "bash")
    process._process = child

    if process_exited:
        assert process.cancel() is False
    else:
        with pytest.raises(ProcessLookupError):
            process.cancel()


def test_unix_cancellation_waits_for_killed_process(tmp_path, monkeypatch):
    kill = MagicMock()
    monkeypatch.setattr(bash_process, "os", SimpleNamespace(name="posix", killpg=kill))
    monkeypatch.setattr(bash_process, "signal", SimpleNamespace(SIGKILL=9))
    child = MagicMock()
    child.pid = 123
    child.poll.return_value = None
    process = BashProcess("sleep 1", tmp_path, "bash")
    process._process = child

    assert process.cancel() is True
    kill.assert_called_once_with(123, 9)
    child.wait.assert_called_once_with()


def test_posix_cancellation_kills_group_and_waits_for_process(tmp_path, monkeypatch):
    kill_group = MagicMock()
    wait = MagicMock()
    child = SimpleNamespace(pid=123, poll=lambda: None, wait=wait)
    monkeypatch.setattr(
        bash_process, "os", SimpleNamespace(name="posix", killpg=kill_group)
    )
    process = BashProcess("printf test", tmp_path, "bash")
    process._process = child
    assert process.cancel() is True
    kill_group.assert_called_once_with(child.pid, bash_process.signal.SIGKILL)
    wait.assert_called_once_with()
