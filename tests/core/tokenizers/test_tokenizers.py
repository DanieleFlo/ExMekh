import io
import json
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from unittest.mock import patch

import pytest

from qs.core.tokenizers.sequential_tokenizer import SequentialTokenizer
from qs.core.tokenizers.tokenizer_session_registry import TokenizerSessionRegistry
from qs.interfaces.tokenizer import Tokenizer
from qs.providers.local import LocalProvider
from qs.tokenizers.qwen import QwenTokenizer
from qs.utils.file import read_json_from_path


class SampleTokenizer(SequentialTokenizer):
    def __init__(self, directory, tokens=("a", "B", "2"), identity="sample"):
        super().__init__(identity, directory)
        self.source = tokens
        self.loads = 0

    def _load_vocabulary_tokens(self):
        self.loads += 1
        return self.source


@pytest.fixture(autouse=True)
def isolated_sessions():
    TokenizerSessionRegistry.close_all()
    yield
    TokenizerSessionRegistry.close_all()


def test_filter_sequence_and_exhaustion(tmp_path):
    tokenizer = SampleTokenizer(
        tmp_path,
        (
            "a",
            "B",
            "2",
            "a",
            "",
            " a",
            "a ",
            "a_b",
            "a-b",
            "é",
            "Ġa",
            "a\n",
            "１",
        ),
    )
    expected = ["_".join(parts) for parts in product(("2", "B", "a"), repeat=3)]
    assert [tokenizer.generate_id() for _ in expected] == expected
    with pytest.raises(RuntimeError, match="exhausted"):
        tokenizer.generate_id()
    assert tokenizer.loads == 1
    counter = next(tmp_path.glob("*/temp/session-*/counter.json"))
    assert read_json_from_path(counter) == {"next_index": 27}


def test_cache_reused_without_source_and_session_shared(tmp_path):
    first = SampleTokenizer(tmp_path)
    assert first.generate_id() == "2_2_2"
    second = SampleTokenizer(tmp_path)
    with patch.object(second, "_load_vocabulary_tokens", side_effect=AssertionError):
        assert second.generate_id() == "2_2_B"
        second.initialize()
    assert first.generate_id() == "2_2_a"


def test_concurrent_instances_do_not_repeat_ids(tmp_path):
    instances = [SampleTokenizer(tmp_path, tuple("abcdef")) for _ in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda n: instances[n % 8].generate_id(), range(200)))
    assert len(set(results)) == 200
    assert sum(instance.loads for instance in instances) == 1


def test_tokenizer_identities_have_separate_paths_and_counters(tmp_path):
    first = SampleTokenizer(tmp_path, identity="vendor/model-one:revision")
    second = SampleTokenizer(tmp_path, identity="vendor/model-two:revision")
    assert first.generate_id() == second.generate_id()
    assert len(list(tmp_path.glob("*/vocabulary.json"))) == 2
    assert len(list(tmp_path.glob("*/temp/session-*/counter.json"))) == 2


def test_new_process_session_reuses_cache_but_restarts_sequence(tmp_path):
    tokenizer = SampleTokenizer(tmp_path)
    assert tokenizer.generate_id() == "2_2_2"
    with patch("qs.core.cache.session_pool.os.getpid", return_value=-1):
        assert tokenizer.generate_id() == "2_2_2"
    assert tokenizer.loads == 1
    assert len(list(tmp_path.glob("*/temp/session-*"))) == 2


def test_failed_counter_write_does_not_reissue_reserved_id(tmp_path):
    tokenizer = SampleTokenizer(tmp_path)
    tokenizer.initialize()
    with (
        patch(
            "qs.core.cache.session_cache.write_json_atomically",
            side_effect=OSError,
        ),
        pytest.raises(OSError),
    ):
        tokenizer.generate_id()
    assert tokenizer.generate_id() == "2_2_B"


@pytest.mark.parametrize("tokens", [[], ["_", " ", "é"]])
def test_empty_vocabulary_not_cached(tmp_path, tokens):
    with pytest.raises(ValueError, match="no ASCII"):
        SampleTokenizer(tmp_path, tokens).generate_id()
    assert not list(tmp_path.rglob("vocabulary.json"))


@pytest.mark.parametrize(
    "data",
    [
        "{",
        "[]",
        '{"version": 2}',
        '{"version": 1, "identity": "wrong"}',
        *[
            json.dumps({"version": 1, "identity": "sample", "tokens": tokens})
            for tokens in [None, [], [1], ["a_b"], ["a", "a"], ["b", "a"]]
        ],
    ],
)
def test_invalid_cache_fails_without_download(tmp_path, data):
    tokenizer = SampleTokenizer(tmp_path)
    tokenizer.initialize()
    cache = next(tmp_path.glob("*/vocabulary.json"))
    cache.write_text(data, encoding="utf-8")
    other = SampleTokenizer(tmp_path)
    with pytest.raises(ValueError, match="Invalid tokenizer cache"):
        other.generate_id()
    assert other.loads == 0


@pytest.mark.parametrize(
    ("identity", "directory", "error"),
    [
        (None, None, TypeError),
        (" ", None, ValueError),
        ("sample", "path", TypeError),
    ],
)
def test_invalid_inputs(identity, directory, error):
    with pytest.raises(error):
        SampleTokenizer(directory, identity=identity)


def test_interfaces_are_abstract():
    with pytest.raises(TypeError):
        Tokenizer()
    with pytest.raises(TypeError):
        SequentialTokenizer("sample")


def test_qwen_loads_official_vocabulary_and_excludes_special_tokens(tmp_path):
    payload = {
        "model": {"vocab": {"word": 0, "Ġword": 1, "special": 2}},
        "added_tokens": [
            {"content": "special", "special": True},
            {"content": "word", "special": False},
        ],
    }
    tokenizer = QwenTokenizer(tmp_path)
    with patch(
        "qs.tokenizers.qwen.urlopen",
        return_value=io.BytesIO(
            json.dumps(payload).encode(),
        ),
    ) as download:
        assert tokenizer.generate_id() == "word_word_word"
    download.assert_called_once_with(
        "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/tokenizer.json",
        timeout=60,
    )
    with patch("qs.tokenizers.qwen.urlopen", side_effect=AssertionError):
        QwenTokenizer(tmp_path).initialize()


def test_download_failure_leaves_no_cache_and_can_retry(tmp_path):
    tokenizer = QwenTokenizer(tmp_path)
    with (
        patch("qs.tokenizers.qwen.urlopen", side_effect=OSError),
        pytest.raises(OSError),
    ):
        tokenizer.initialize()
    assert not list(tmp_path.rglob("vocabulary.json"))
    with patch.object(tokenizer, "_load_vocabulary_tokens", return_value=["retry"]):
        assert tokenizer.generate_id() == "retry_retry_retry"


def test_provider_accepts_tokenizer_and_delegates(tmp_path):
    tokenizer = SampleTokenizer(tmp_path)
    with (
        patch.object(LocalProvider, "_build_client"),
        patch.object(
            LocalProvider,
            "_check_provider",
        ),
    ):
        first = LocalProvider("model", tokenizer=tokenizer)
        second = LocalProvider("model", tokenizer=SampleTokenizer(tmp_path))
        default = LocalProvider("model")
    assert first.get_tokenizer() is tokenizer
    assert isinstance(default.get_tokenizer(), QwenTokenizer)
    assert first.generate_id() == "2_2_2"
    assert second.generate_id() == "2_2_B"
    with pytest.raises(TypeError, match="Tokenizer"):
        LocalProvider("model", tokenizer=object())
