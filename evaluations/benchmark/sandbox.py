"""Bounded Python execution with Linux namespaces, Landlock and seccomp."""

import json
import os
import shutil
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any


@dataclass(frozen=True)
class Execution:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    infrastructure_error: str | None = None
    output_truncated: bool = False

    @property
    def succeeded(self) -> bool:
        return (
            self.exit_code == 0
            and not self.timed_out
            and not self.output_truncated
            and self.infrastructure_error is None
        )


def workspace_path(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(
        root.resolve()
    ):
        raise ValueError("Path must stay inside the trial workspace")
    if any(
        part.is_symlink() for part in (path, *path.parents) if part.is_relative_to(root)
    ):
        raise ValueError("Workspace symlinks are not accepted")
    return path


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # The process may finish between timeout/interruption and signalling.
    process.communicate()


def execute(
    workspace: Path, request: dict[str, Any], *, timeout: float = 8
) -> Execution:
    if sys.platform != "linux" or not shutil.which("unshare"):
        return Execution(
            None,
            "",
            "",
            infrastructure_error="Linux unshare is required; code was not executed",
        )
    workspace_path(workspace, request["path"])
    runtime = Path(sys.base_prefix).resolve()
    with TemporaryDirectory(prefix="agent-benchmark-sandbox-") as temporary:
        scratch = Path(temporary)
        (scratch / "temporary").mkdir()
        ready_read, ready_write = os.pipe()
        command = [
            "unshare",
            "--user",
            "--map-root-user",
            "--net",
            "--pid",
            "--fork",
            "--kill-child",
            sys.executable,
            "-I",
            "-S",
            str(Path(__file__).with_name("sandbox_worker.py")),
            str(workspace.resolve()),
            str(scratch / "temporary"),
            str(runtime),
            str(ready_write),
        ]
        with (
            os.fdopen(ready_read, "rb") as readiness,
            os.fdopen(ready_write, "wb") as ready_sender,
            (scratch / "stdout").open("w+b") as stdout,
            (scratch / "stderr").open("w+b") as stderr,
        ):
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
                env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin"},
                start_new_session=True,
                pass_fds=(ready_sender.fileno(),),
            )
            ready_sender.close()
            timed_out = False
            try:
                process.communicate(json.dumps(request).encode(), timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_process(process)
            except BaseException:
                _terminate_process(process)
                raise
            ready = readiness.read(1) == b"1"
            output_truncated = stdout.tell() > 65_536 or stderr.tell() > 65_536
            stdout.seek(0)
            stderr.seek(0)
            text = stdout.read(65_536).decode("utf-8", errors="replace")
            errors = stderr.read(65_536).decode("utf-8", errors="replace")
        setup_failed = not ready and not timed_out
        return Execution(
            process.returncode,
            text,
            errors,
            timed_out,
            "Namespace setup failed: " + errors if setup_failed else None,
            output_truncated,
        )


def run_python(
    work_root: Path,
    path: Annotated[str, "Workroot-relative Python script; standard library only."],
    arguments: Annotated[
        list[str] | None, "Command-line arguments, not shell syntax; null means none."
    ] = None,
) -> dict[str, Any]:
    """Run a Python script without network or host-home access; return exit/stdout/stderr."""
    return asdict(
        execute(
            work_root,
            {"path": path, "mode": "script", "arguments": list(arguments or ())},
        )
    )
