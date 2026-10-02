from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.request import urlopen

from tokenizers import Encoding
from tokenizers import Tokenizer as TextTokenizer

from exmekh_lib.config.tokenizer import QWEN_TOKENIZER, TokenizerSettings
from exmekh_lib.core.tokenizers.sequential_tokenizer import SequentialTokenizer
from exmekh_lib.utils.file import read_json_from_path, write_json_atomically


class QwenTokenizer(SequentialTokenizer):
    MODEL = QWEN_TOKENIZER.repository

    def __init__(self, cache_directory: Path | None = None) -> None:
        super().__init__(QWEN_TOKENIZER.identity, cache_directory)
        self._text_tokenizer: TextTokenizer | None = None
        self._text_tokenizer_lock = Lock()
        self._tokenizer_path = self._directory / QWEN_TOKENIZER.filename

    def _load_vocabulary_tokens(self) -> Iterable[str]:
        data = read_json_from_path(self._ensure_tokenizer_file())
        vocabulary = data["model"]["vocab"]
        special_tokens = {
            token["content"]
            for token in data.get("added_tokens", [])
            if token.get("special", False)
        }
        return (token for token in vocabulary if token not in special_tokens)

    def _download_tokenizer_data(self) -> dict[str, Any]:
        with urlopen(
            QWEN_TOKENIZER.download_url,
            timeout=TokenizerSettings.from_environment().download_timeout_seconds,
        ) as response:
            return json.load(response)

    def _ensure_tokenizer_file(self) -> Path:
        if not self._tokenizer_path.exists():
            data = self._download_tokenizer_data()
            self._directory.mkdir(parents=True, exist_ok=True)
            write_json_atomically(self._tokenizer_path, data)
        return self._tokenizer_path

    def _get_text_tokenizer(self) -> TextTokenizer:
        with self._text_tokenizer_lock:
            tokenizer = self._text_tokenizer
            if tokenizer is None:
                self.initialize()
                tokenizer = TextTokenizer.from_file(str(self._ensure_tokenizer_file()))
                self._text_tokenizer = tokenizer
            return tokenizer

    def count_tokens(self, text: str) -> int:
        return len(self._encode(text).ids)

    def truncate_text(self, text: str, start_tokens: int, end_tokens: int = 0) -> str:
        self._validate_token_limit(start_tokens, "start_tokens")
        self._validate_token_limit(end_tokens, "end_tokens")

        encoding = self._encode(text)
        token_budget = start_tokens + end_tokens
        if len(encoding.ids) <= token_budget:
            return text

        offsets = encoding.offsets
        # Token offsets can share a character, and joining both ends can
        # change tokenization; verify the resulting text against the budget.
        while start_tokens or end_tokens:
            prefix_end = offsets[start_tokens - 1][1] if start_tokens else 0
            suffix_start = offsets[-end_tokens][0] if end_tokens else len(text)
            result = text[:prefix_end] + text[suffix_start:]
            if self.count_tokens(result) <= token_budget:
                return result
            if start_tokens >= end_tokens and start_tokens:
                start_tokens -= 1
            else:
                end_tokens -= 1
        return ""

    def _encode(self, text: str) -> Encoding:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return self._get_text_tokenizer().encode(text, add_special_tokens=False)

    @staticmethod
    def _validate_token_limit(count: int, name: str) -> None:
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"{name} must be an integer")
        if count < 0:
            raise ValueError(f"{name} cannot be negative")
