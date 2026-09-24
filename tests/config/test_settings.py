import io
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import pytest

from qs.config.cli import CliSettings
from qs.config.provider import ProviderSettings
from qs.config.runtime import RUNTIME
from qs.config.storage import StorageSettings
from qs.config.tokenizer import QWEN_TOKENIZER, TokenizerSettings
from qs.core.cache.cache_manager import CacheManager
from qs.core.tokenizers.sequential_tokenizer import SequentialTokenizer
from qs.tokenizers.qwen import QwenTokenizer


class StubTokenizer(SequentialTokenizer):
    def _load_vocabulary_tokens(self):
        return ["token"]

    def count_tokens(self, text: str) -> int:
        return len(text)

    def truncate_text(self, text: str, start_tokens: int, end_tokens: int = 0) -> str:
        return text[:start_tokens] + (text[-end_tokens:] if end_tokens else "")


def test_provider_settings_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QS_MODEL", "custom-model")
    monkeypatch.setenv("QS_API", "chat_completions")

    settings = ProviderSettings.from_environment()

    assert settings.model == "custom-model"
    assert settings.api == "chat_completions"
    assert settings.api_key_env_var == "API_KEY"


def test_cli_settings_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QS_ROOT_HDS", "custom-root")
    monkeypatch.setenv("QS_CHILD_HDS", "custom-child")

    assert CliSettings.from_environment() == CliSettings(
        root_hds="custom-root", child_hds="custom-child"
    )


def test_storage_setting_is_shared_by_cache_and_tokenizer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("QS_TOKENIZER_CACHE_DIR", str(tmp_path))

    assert StorageSettings.from_environment().tokenizer_cache_directory == tmp_path
    assert CacheManager.resolve_root() == tmp_path.resolve()
    tokenizer = StubTokenizer("sample")
    assert tokenizer._directory.parent == tmp_path.resolve()


def test_internal_settings_are_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        RUNTIME.required_tool_choice = "optional"
    with pytest.raises(FrozenInstanceError):
        QWEN_TOKENIZER.repository = "another/model"
    assert QWEN_TOKENIZER.identity == CacheManager.DEFAULT_TOKENIZER_IDENTITY


def test_tokenizer_timeout_can_be_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("QS_TOKENIZER_DOWNLOAD_TIMEOUT", "7")

    assert TokenizerSettings.from_environment().download_timeout_seconds == 7
    with patch("qs.tokenizers.qwen.urlopen", return_value=io.BytesIO(b"{}")) as fetch:
        QwenTokenizer(tmp_path)._download_tokenizer_data()
    fetch.assert_called_once_with(QWEN_TOKENIZER.download_url, timeout=7)


@pytest.mark.parametrize("value", ["invalid", "0", "-1"])
def test_tokenizer_timeout_requires_positive_integer(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("QS_TOKENIZER_DOWNLOAD_TIMEOUT", value)

    with pytest.raises(ValueError, match="positive integer"):
        TokenizerSettings.from_environment()
