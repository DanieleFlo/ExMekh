from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from pathlib import Path

from exmekh_lib.config.cache import CACHE_FORMAT
from exmekh_lib.config.validation import VALIDATION_PATTERNS
from exmekh_lib.utils.file import read_json_from_path, write_json_atomically

logger = logging.getLogger(__name__)


class VocabularyCache:
    _TOKEN_PATTERN = re.compile(VALIDATION_PATTERNS.ascii_vocabulary_token)

    def __init__(self, directory: Path, identity: str) -> None:
        self._directory = directory
        self._identity = identity
        self._path = directory / CACHE_FORMAT.vocabulary_filename

    def load_or_build(
        self,
        load_vocabulary_tokens: Callable[[], Iterable[str]],
    ) -> tuple[str, ...]:
        if self._path.exists():
            tokens = self._read_cached_vocabulary()
            logger.info(
                "Loaded cached vocabulary: identity=%s tokens=%d path=%s",
                self._identity,
                len(tokens),
                self._path,
            )
            return tokens
        tokens = tuple(
            sorted(
                {
                    token
                    for token in load_vocabulary_tokens()
                    if self._TOKEN_PATTERN.fullmatch(token)
                }
            )
        )
        if not tokens:
            raise ValueError("Tokenizer has no ASCII alphanumeric tokens")
        self._directory.mkdir(parents=True, exist_ok=True)
        value = {
            "version": 1,
            "identity": self._identity,
            "tokens": tokens,
        }
        write_json_atomically(self._path, value)
        logger.info(
            "Built and cached vocabulary: identity=%s tokens=%d path=%s",
            self._identity,
            len(tokens),
            self._path,
        )
        return tokens

    def _read_cached_vocabulary(self) -> tuple[str, ...]:
        try:
            data = read_json_from_path(self._path)
            if (
                not isinstance(data, dict)
                or data.get("version") != 1
                or data.get("identity") != self._identity
            ):
                raise ValueError("Invalid cache metadata")
            tokens = data.get("tokens")
            if (
                not isinstance(tokens, list)
                or not tokens
                or any(
                    not isinstance(token, str)
                    or not self._TOKEN_PATTERN.fullmatch(token)
                    for token in tokens
                )
                or tokens != sorted(set(tokens))
            ):
                raise ValueError("Invalid cached vocabulary")
        except (ValueError, UnicodeError) as error:
            raise ValueError(f"Invalid tokenizer cache: {self._path}") from error
        return tuple(tokens)
