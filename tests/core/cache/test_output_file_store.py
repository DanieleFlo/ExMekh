import os
import subprocess
import sys
from pathlib import Path

import pytest

from exmekh_lib.core.cache import SessionCache
from exmekh_lib.core.cache.output_file_store import OutputFileSession


def test_startup_cleanup_keeps_live_sessions_and_close_only_removes_owned_files(
    tmp_path,
):
    root = tmp_path / "outputs"
    first = OutputFileSession(root)
    second = OutputFileSession(root)
    try:
        first_path = first.write("node", "id", ("first",))[0]
        second_path = second.write("node", "id", ("second",))[0]
        OutputFileSession.clean_abandoned(root)
        assert first_path.read_text() == "first"
        assert second_path.read_text() == "second"
        first.close()
        assert not first_path.exists()
        assert second_path.exists()
        first.close()
    finally:
        first.close()
        second.close()


def test_process_crash_is_cleaned_on_next_startup_without_removing_active_process(
    tmp_path,
):
    root = tmp_path / "outputs"
    script = "from pathlib import Path; from exmekh_lib.core.cache.output_file_store import OutputFileSession; import os; session = OutputFileSession(Path(os.environ['OUTPUT_ROOT'])); print(session.write('node', 'id', ('abandoned',))[0], flush=True); os._exit(0)"
    environment = {
        **os.environ,
        "OUTPUT_ROOT": str(root),
        "PYTHONPATH": str(Path(__file__).resolve().parents[3] / "src"),
    }
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    abandoned = Path(result.stdout.strip())
    assert abandoned.exists()
    active = OutputFileSession(root)
    try:
        assert not abandoned.exists()
        live = active.write("node", "id", ("live",))[0]
        cleanup_script = "from pathlib import Path; from exmekh_lib.core.cache.output_file_store import OutputFileSession; import os; OutputFileSession.clean_abandoned(Path(os.environ['OUTPUT_ROOT']))"
        subprocess.run(
            [sys.executable, "-c", cleanup_script],
            env=environment,
            check=True,
            timeout=10,
        )
        assert live.exists()
    finally:
        active.close()


def test_session_cache_owns_output_files_and_nodes_do_not_collide(tmp_path):
    session = SessionCache(tmp_path / "cache")
    first = session.node("a/b")
    second = session.node("a_b")
    store = session.output_files(tmp_path / "outputs")
    first_path = first.write_output_files(tmp_path / "outputs", "id", ("first",))[0]
    second_path = second.write_output_files(tmp_path / "outputs", "id", ("second",))[0]
    assert first_path != second_path
    session.close()
    assert not first_path.exists() and not second_path.exists()
    with pytest.raises(RuntimeError, match="closed"):
        store.write(first.node_id, "later", ("late",))


def test_cleanup_ignores_unowned_directories_and_symlinks(tmp_path):
    root = tmp_path / "outputs"
    root.mkdir()
    unowned = root / "session-user-data"
    unowned.mkdir()
    (unowned / "keep.txt").write_text("keep")
    other = tmp_path / "other"
    other.mkdir()
    (other / ".lease").write_text("0")
    (other / "keep.txt").write_text("keep")
    (root / "session-link").symlink_to(other, target_is_directory=True)
    OutputFileSession.clean_abandoned(root)
    assert (unowned / "keep.txt").exists()
    assert (other / "keep.txt").exists()


@pytest.mark.parametrize("change", ["overwrite", "remove", "replace", "symlink"])
def test_changed_exports_cannot_be_reused_or_granted_an_expanded_read_limit(
    tmp_path, change
):
    with SessionCache(tmp_path / "cache") as session:
        cache = session.node("owner")
        root = tmp_path / "outputs"
        path = cache.write_output_files(root, "id", ("original",))[0]
        if change == "overwrite":
            path.write_text("modified")
        else:
            path.unlink()
            if change == "replace":
                path.write_text("original")
            elif change == "symlink":
                other = tmp_path / "other.txt"
                other.write_text("original")
                path.symlink_to(other)
        with pytest.raises(OSError, match="unavailable"):
            cache.write_output_files(root, "id", ("original",))
        with pytest.raises(ValueError, match="not available"):
            cache.output_file_read_limit(root, path, 1)


def test_owned_output_read_preserves_bytes_and_rejects_other_nodes_or_closed_sessions(
    tmp_path,
):
    root = tmp_path / "outputs"
    store = OutputFileSession(root)
    text = "first\r\n\nlast\ud800"
    path = store.write("owner", "result", (text,))[0]
    try:
        assert store.read_text("owner", path) == text
        for node_id, candidate in (
            ("other", path),
            ("owner", tmp_path / "unregistered"),
        ):
            with pytest.raises(ValueError, match="not available"):
                store.read_text(node_id, candidate)
    finally:
        store.close()
    with pytest.raises(ValueError, match="not available"):
        store.read_text("owner", path)


@pytest.mark.parametrize(
    "mutation", ["before_open", "during_read", "replace_after_read"]
)
def test_output_read_rejects_a_snapshot_changed_during_access(
    tmp_path, monkeypatch, mutation
):
    store = OutputFileSession(tmp_path / "outputs")
    path = store.write("owner", "result", ("original",))[0]
    original_open = Path.open

    class ChangingReader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def fileno(self):
            return self.stream.fileno()

        def read(self):
            data = self.stream.read()
            if mutation == "during_read":
                descriptor = os.open(path, os.O_WRONLY | os.O_TRUNC)
                try:
                    os.write(descriptor, b"changed")
                finally:
                    os.close(descriptor)
            else:
                path.unlink()
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT)
                try:
                    os.write(descriptor, b"original")
                finally:
                    os.close(descriptor)
            return data

    def open_with_change(candidate, *args, **kwargs):
        if candidate == path and mutation == "before_open":
            descriptor = os.open(path, os.O_WRONLY | os.O_TRUNC)
            try:
                os.write(descriptor, b"changed")
            finally:
                os.close(descriptor)
        stream = original_open(candidate, *args, **kwargs)
        return ChangingReader(stream) if candidate == path else stream

    monkeypatch.setattr(Path, "open", open_with_change)
    try:
        with pytest.raises(ValueError, match="not available"):
            store.read_text("owner", path)
    finally:
        store.close()
