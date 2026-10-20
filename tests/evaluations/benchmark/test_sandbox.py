"""Generated Python is checked in a bounded process with no unrestricted fallback."""

import json
from unittest.mock import MagicMock

import pytest

from evaluations.benchmark.sandbox import execute, workspace_path


def run_source(tmp_path, source, **options):
    (tmp_path / "candidate.py").write_text(source)
    result = execute(
        tmp_path, {"path": "candidate.py", "mode": "script", "arguments": []}, **options
    )
    if result.infrastructure_error:
        pytest.skip(result.infrastructure_error)
    return result


def test_standard_library_script_executes(tmp_path):
    result = run_source(tmp_path, "import math\nprint(math.factorial(5))")
    assert result.exit_code == 0
    assert result.stdout.strip() == "120"


def test_network_process_execution_and_workspace_mutations_are_denied(tmp_path):
    source = """import json, os, socket, subprocess
results = []
for action in [lambda: open('candidate.py', 'w'), lambda: socket.socket(), lambda: subprocess.run(['/usr/bin/true']), lambda: os.chmod('candidate.py', 0o777)]:
    try:
        action()
        results.append(False)
    except OSError:
        results.append(True)
print(json.dumps(results))
"""
    result = run_source(tmp_path, source)
    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout) == [True] * 4
    assert (tmp_path / "candidate.py").read_text() == source


def test_host_content_outside_workspace_is_unreadable(tmp_path):
    secret = tmp_path.parent / (tmp_path.name + "-secret")
    secret.write_text("PRIVATE")
    result = run_source(tmp_path, f"open({str(secret)!r}).read()")
    assert result.exit_code != 0
    assert "PermissionError" in result.stderr


def test_timeout_and_output_limits_are_visible(tmp_path):
    result = run_source(tmp_path, "import time\ntime.sleep(10)", timeout=0.5)
    assert result.timed_out
    result = run_source(tmp_path, "print('a' * 70000)")
    assert result.output_truncated


def test_paths_cannot_escape_or_use_symlinks(tmp_path):
    with pytest.raises(ValueError):
        workspace_path(tmp_path, "../escape.py")
    (tmp_path / "link.py").symlink_to(tmp_path / "candidate.py")
    with pytest.raises(ValueError):
        workspace_path(tmp_path, "link.py")


def test_candidate_stderr_cannot_impersonate_a_sandbox_setup_failure(tmp_path):
    run_source(tmp_path, "print('ready')")
    (tmp_path / "candidate.py").write_text(
        "import sys\nprint('BENCHMARK_SANDBOX_UNAVAILABLE: unshare:', file=sys.stderr)\nraise SystemExit(125)"
    )
    result = execute(
        tmp_path, {"path": "candidate.py", "mode": "script", "arguments": []}
    )
    assert result.infrastructure_error is None
    assert result.exit_code == 125
    assert not result.succeeded


@pytest.mark.parametrize("already_finished", [False, True])
def test_interruption_terminates_the_program_before_propagating(
    tmp_path, monkeypatch, already_finished
):
    process = MagicMock(pid=12345)
    process.communicate.side_effect = [KeyboardInterrupt, None]
    monkeypatch.setattr(
        "evaluations.benchmark.sandbox.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    kill = MagicMock(side_effect=ProcessLookupError if already_finished else None)
    monkeypatch.setattr("evaluations.benchmark.sandbox.os.killpg", kill)
    with pytest.raises(KeyboardInterrupt):
        execute(tmp_path, {"path": "candidate.py", "mode": "script", "arguments": []})
    kill.assert_called_once()
    assert process.communicate.call_count == 2
