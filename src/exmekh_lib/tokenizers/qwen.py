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
from exmekh_lib.config.tokenizer_text import TOKENIZER_TEXT
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

    def split_text(self, text: str, max_tokens: int) -> tuple[str, ...]:
        """Encode the source once, then verify only bounded candidate chunks."""
        self._validate_token_limit(max_tokens, "max_tokens")
        if max_tokens == 0:
            raise ValueError(TOKENIZER_TEXT.max_tokens_must_be_positive)
        encoding = self._encode(text)
        if len(encoding.ids) <= max_tokens:
            return (text,) if text else ()
        offsets = encoding.offsets
        token_start = 0
        character_start = 0
        parts: list[str] = []
        while token_start < len(offsets):
            token_end = min(token_start + max_tokens, len(offsets))
            part = ""
            character_end = character_start
            while token_end > token_start:
                character_end = (
                    len(text)
                    if token_end == len(offsets)
                    else offsets[token_end - 1][1]
                )
                part = text[character_start:character_end]
                if part and self.count_tokens(part) <= max_tokens:
                    break
                token_end -= 1
            if token_end == token_start:
                raise ValueError(
                    TOKENIZER_TEXT.tokenizer_cannot_split_a_character_within_the_token_budget
                )
            parts.append(part)
            character_start = character_end
            token_start = token_end
            # Byte-level tokens can share the same Unicode character offset.
            while (
                token_start < len(offsets)
                and offsets[token_start][1] <= character_start
            ):
                token_start += 1
        if character_start < len(text):
            raise ValueError(
                TOKENIZER_TEXT.tokenizer_did_not_cover_the_complete_source_text
            )
        return tuple(parts)

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
            raise TypeError(TOKENIZER_TEXT.text_must_be_a_string)
        return self._get_text_tokenizer().encode(text, add_special_tokens=False)
