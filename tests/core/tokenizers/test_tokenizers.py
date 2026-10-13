import io
import json
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from tokenizers import Tokenizer as TextTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers

from exmekh_lib.config.cache import CACHE_FORMAT
from exmekh_lib.config.tokenizer import QWEN_TOKENIZER, TokenizerSettings
from exmekh_lib.core.tokenizers.sequential_tokenizer import SequentialTokenizer
from exmekh_lib.core.tokenizers.tokenizer_session_registry import (
    TokenizerSessionRegistry,
)
from exmekh_lib.interfaces.tokenizer import Tokenizer
from exmekh_lib.providers.local import LocalProvider
from exmekh_lib.tokenizers.qwen import QwenTokenizer
from exmekh_lib.utils.file import read_json_from_path


class SampleTokenizer(SequentialTokenizer):
    def count_tokens(self, text: str) -> int:
        return len(text)

    def truncate_text(self, text: str, start_tokens: int, end_tokens: int = 0) -> str:
        return text[:start_tokens] + (text[-end_tokens:] if end_tokens else "")

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
    counter = next(
        tmp_path.glob(
            f"*/{CACHE_FORMAT.temporary_folder}/{CACHE_FORMAT.session_prefix}*/{CACHE_FORMAT.counter_filename}"
        )
    )
    assert read_json_from_path(counter) == {"next_index": 27}


def test_cache_reused_without_source_and_session_shared(tmp_path):
    first = SampleTokenizer(tmp_path)
    assert first.generate_id() == "2_2_2"
    second = SampleTokenizer(tmp_path)
    with patch.object(second, "_load_vocabulary_tokens", side_effect=AssertionError):
        assert second.generate_id() == "2_2_B"
        second.initialize()
    assert first.generate_id() == "2_2_a"


def test_id_vocabulary_is_filtered_reused_and_does_not_reserve_ids(tmp_path):
    tokenizer = SampleTokenizer(tmp_path, ("a", "B", "2", "a", " a", "a_b"))
    vocabulary = tokenizer.get_id_vocabulary()
    assert vocabulary == ("2", "B", "a")
    assert tokenizer.get_id_vocabulary() is vocabulary
    assert tokenizer.loads == 1
    assert not list(tmp_path.glob(f"*/{CACHE_FORMAT.temporary_folder}"))
    assert tokenizer.generate_id() == "2_2_2"


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
    assert len(list(tmp_path.glob(f"*/{CACHE_FORMAT.vocabulary_filename}"))) == 2
    assert (
        len(
            list(
                tmp_path.glob(
                    f"*/{CACHE_FORMAT.temporary_folder}/{CACHE_FORMAT.session_prefix}*/{CACHE_FORMAT.counter_filename}"
                )
            )
        )
        == 2
    )


def test_new_process_session_reuses_cache_but_restarts_sequence(tmp_path):
    tokenizer = SampleTokenizer(tmp_path)
    assert tokenizer.generate_id() == "2_2_2"
    with patch("exmekh_lib.core.cache.session_pool.os.getpid", return_value=-1):
        assert tokenizer.generate_id() == "2_2_2"
    assert tokenizer.loads == 1
    assert (
        len(
            list(
                tmp_path.glob(
                    f"*/{CACHE_FORMAT.temporary_folder}/{CACHE_FORMAT.session_prefix}*"
                )
            )
        )
        == 2
    )


def test_failed_counter_write_does_not_reissue_reserved_id(tmp_path):
    tokenizer = SampleTokenizer(tmp_path)
    tokenizer.initialize()
    with (
        patch(
            "exmekh_lib.core.cache.session_cache.write_json_atomically",
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
    assert not list(tmp_path.rglob(CACHE_FORMAT.vocabulary_filename))


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
    cache = next(tmp_path.glob(f"*/{CACHE_FORMAT.vocabulary_filename}"))
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
        "exmekh_lib.tokenizers.qwen.urlopen",
        return_value=io.BytesIO(
            json.dumps(payload).encode(),
        ),
    ) as download:
        assert tokenizer.generate_id() == "word_word_word"
    download.assert_called_once_with(
        QWEN_TOKENIZER.download_url,
        timeout=TokenizerSettings.from_environment().download_timeout_seconds,
    )
    with patch("exmekh_lib.tokenizers.qwen.urlopen", side_effect=AssertionError):
        QwenTokenizer(tmp_path).initialize()


def _sample_text_tokenizer() -> TextTokenizer:
    tokenizer = TextTokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        ["alpha beta gamma delta café end"],
        trainer=trainers.BpeTrainer(
            vocab_size=300,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["[UNK]"],
        ),
    )
    return tokenizer


def test_qwen_counts_and_truncates_model_tokens(tmp_path):
    tokenizer = QwenTokenizer(tmp_path)
    tokenizer._text_tokenizer = _sample_text_tokenizer()
    text = "alpha café beta gamma delta end " * 30

    assert tokenizer.count_tokens("") == 0
    assert tokenizer.count_tokens(text) > 100
    assert tokenizer.truncate_text(text, tokenizer.count_tokens(text)) == text
    prefix = tokenizer.truncate_text(text, 100)
    assert text.startswith(prefix)
    assert tokenizer.count_tokens(prefix) <= 100

    both_ends = tokenizer.truncate_text(text, 20, 20)
    assert text.startswith(both_ends[:5])
    assert text.endswith(both_ends[-5:])
    assert tokenizer.count_tokens(both_ends) <= 40
    suffix = tokenizer.truncate_text(text, 0, 20)
    assert text.endswith(suffix)
    assert tokenizer.count_tokens(suffix) <= 20
    assert tokenizer.truncate_text(text, 0, 0) == ""


def test_qwen_keeps_the_requested_budget_when_a_prefix_retokenizes(
    tmp_path, monkeypatch
):
    tokenizer = QwenTokenizer(tmp_path)

    def encode(text):
        counts = {"abcdef": 6, "abc": 4, "ab": 3}
        return SimpleNamespace(
            ids=list(range(counts[text])),
            offsets=[(index, index + 1) for index in range(len(text))],
        )

    monkeypatch.setattr(tokenizer, "_encode", encode)
    assert tokenizer.truncate_text("abcdef", 3) == "ab"


def test_qwen_keeps_the_requested_budget_when_a_suffix_retokenizes(
    tmp_path, monkeypatch
):
    tokenizer = QwenTokenizer(tmp_path)

    def encode(text):
        counts = {"abcdef": 6, "aef": 4, "af": 2}
        return SimpleNamespace(
            ids=list(range(counts[text])),
            offsets=[(index, index + 1) for index in range(len(text))],
        )

    monkeypatch.setattr(tokenizer, "_encode", encode)
    assert tokenizer.truncate_text("abcdef", 1, 2) == "af"


@pytest.mark.parametrize(
    ("text", "start", "end", "error"),
    [
        (None, 1, 0, TypeError),
        ("text", True, 0, TypeError),
        ("text", 1, 1.5, TypeError),
        ("text", -1, 0, ValueError),
        ("text", 1, -1, ValueError),
    ],
)
def test_qwen_rejects_invalid_truncation_inputs(tmp_path, text, start, end, error):
    tokenizer = QwenTokenizer(tmp_path)
    tokenizer._text_tokenizer = _sample_text_tokenizer()
    with pytest.raises(error):
        tokenizer.truncate_text(text, start, end)


def test_qwen_rejects_non_string_text_for_counting(tmp_path):
    with pytest.raises(TypeError, match="text must be a string"):
        QwenTokenizer(tmp_path).count_tokens(None)


def test_qwen_reuses_cached_full_tokenizer_for_text(tmp_path):
    payload = json.loads(_sample_text_tokenizer().to_str())
    with patch(
        "exmekh_lib.tokenizers.qwen.urlopen",
        return_value=io.BytesIO(json.dumps(payload).encode()),
    ) as download:
        first = QwenTokenizer(tmp_path)
        assert first.count_tokens("alpha café") > 0
    download.assert_called_once()

    with patch("exmekh_lib.tokenizers.qwen.urlopen", side_effect=AssertionError):
        second = QwenTokenizer(tmp_path)
        assert second.count_tokens("alpha café") == first.count_tokens("alpha café")


def test_qwen_downloads_full_tokenizer_for_an_existing_vocabulary_cache(tmp_path):
    payload = json.loads(_sample_text_tokenizer().to_str())
    tokenizer = QwenTokenizer(tmp_path)
    with patch(
        "exmekh_lib.tokenizers.qwen.urlopen",
        return_value=io.BytesIO(json.dumps(payload).encode()),
    ):
        tokenizer.generate_id()
    (tokenizer._directory / QWEN_TOKENIZER.filename).unlink()

    with patch(
        "exmekh_lib.tokenizers.qwen.urlopen",
        return_value=io.BytesIO(json.dumps(payload).encode()),
    ) as download:
        assert QwenTokenizer(tmp_path).count_tokens("alpha café") > 0
    download.assert_called_once()


def test_download_failure_leaves_no_cache_and_can_retry(tmp_path):
    tokenizer = QwenTokenizer(tmp_path)
    with (
        patch("exmekh_lib.tokenizers.qwen.urlopen", side_effect=OSError),
        pytest.raises(OSError),
    ):
        tokenizer.initialize()
    assert not list(tmp_path.rglob(CACHE_FORMAT.vocabulary_filename))
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
