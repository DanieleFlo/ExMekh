"""Run and monitor one cancellable Bash process."""

from __future__ import annotations

import codecs
import os
import shutil
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from typing import IO, Any

import psutil

from exmekh_lib.config.runtime import RUNTIME


class BashProcess:
    def __init__(self, command: str, work_root: Path, executable: str | None) -> None:
        self._command = command
        self._work_root = work_root
        self._executable = executable
        self._lock = Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._cancel_requested = False
        self._progress: list[str] = []
        self._progress_length = 0
        self._stdout: list[str] = []
        self._stderr: list[str] = []
        self._read_errors: list[OSError] = []

    def run(self) -> dict[str, Any]:
        executable = self._executable or shutil.which("bash")
        if executable is None:
            raise FileNotFoundError(RUNTIME.bash_unavailable)
        started_at = datetime.now(timezone.utc)
        with self._lock:
            if self._cancel_requested:
                raise RuntimeError(RUNTIME.bash_cancelled_before_launch)
            self._process = subprocess.Popen(
                [executable, "-c", self._command],
                cwd=self._work_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                start_new_session=os.name != "nt",
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                ),
            )
            process = self._process
        stdout = process.stdout
        stderr = process.stderr
        if stdout is None or stderr is None:
            self.cancel()
            raise RuntimeError(RUNTIME.bash_pipes_required)
        readers = (
            Thread(target=self._drain, args=(stdout, self._stdout, "stdout")),
            Thread(target=self._drain, args=(stderr, self._stderr, "stderr")),
        )
        for reader in readers:
            reader.start()
        exit_code = process.wait()
        for reader in readers:
            reader.join()
        if self._read_errors:
            raise self._read_errors[0]
        return {
            "command": self._command,
            "cwd": str(self._work_root),
            "status": "exited",
            "exit_code": exit_code,
            "stdout": "".join(self._stdout),
            "stderr": "".join(self._stderr),
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }

    def _drain(self, pipe: IO[bytes], chunks: list[str], stream: str) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            while raw := os.read(pipe.fileno(), 4096):
                self._append(chunks, stream, decoder.decode(raw))
            self._append(chunks, stream, decoder.decode(b"", final=True))
        except OSError as error:
            with self._lock:
                self._read_errors.append(error)
        finally:
            pipe.close()

    def _append(self, chunks: list[str], stream: str, content: str) -> None:
        if content:
            with self._lock:
                chunks.append(content)
                progress = f"[{stream}] {content}"
                self._progress.append(progress)
                self._progress_length += len(progress)

    def read_partial(self, cursor: int) -> tuple[str, int]:
        with self._lock:
            return "".join(self._progress)[cursor:], self._progress_length

    def cancel(self) -> bool:
        with self._lock:
            process = self._process
            if process is None:
                self._cancel_requested = True
                return True
            if process.poll() is not None:
                return False
        if os.name == "nt":
            if not self._kill_windows_tree(process):
                return False
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                if process.poll() is None:
                    raise
                return False
        process.wait()
        return True

    @staticmethod
    def _kill_windows_tree(process: subprocess.Popen[bytes]) -> bool:
        try:
            parent = psutil.Process(process.pid)
            descendants = parent.children(recursive=True)
            for child in reversed(descendants):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            try:
                parent.kill()
            except psutil.NoSuchProcess:
                pass
        except psutil.NoSuchProcess:
            if process.poll() is None:
                raise
            return False
        return True
