import pytest

from qs.core.tokenizers.tokenizer_session import TokenizerSession
from qs.utils.file import read_json_from_path


def test_session_close_removes_storage_and_prevents_reuse(tmp_path):
    session = TokenizerSession(tmp_path)
    assert session.reserve_next_index(10) == 0
    counter = next(tmp_path.glob("session-*/counter.json"))
    assert read_json_from_path(counter) == {"next_index": 1}

    session.close()
    session.close()

    assert not list(tmp_path.iterdir())
    with pytest.raises(RuntimeError, match="closed"):
        session.reserve_next_index(10)
