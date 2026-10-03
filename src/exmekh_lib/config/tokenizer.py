from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QwenTokenizerSpec:
    """Fixed vocabulary identity and download details for QwenTokenizer."""

    repository: str = "Qwen/Qwen3.8-27B"
    download_base_url: str = "https://huggingface.co"
    revision: str = "main"
    vocabulary_version: str = "ascii-v1"
    filename: str = "tokenizer.json"

    @property
    def identity(self) -> str:
        return f"{self.repository}:{self.revision}:{self.vocabulary_version}"

    @property
    def download_url(self) -> str:
        return (
            f"{self.download_base_url}/{self.repository}/resolve/"
            f"{self.revision}/{self.filename}"
        )


QWEN_TOKENIZER = QwenTokenizerSpec()


@dataclass(frozen=True, slots=True)
class TokenizerSettings:
    """User-facing network timeout for downloading the tokenizer."""

    download_timeout_seconds: int = 60

    @classmethod
    def from_environment(cls) -> TokenizerSettings:
        value = os.getenv("EXMEKH_LIB_TOKENIZER_DOWNLOAD_TIMEOUT")
        if value is None:
            return cls()
        try:
            timeout = int(value)
        except ValueError as error:
            raise ValueError(
                "EXMEKH_LIB_TOKENIZER_DOWNLOAD_TIMEOUT must be a positive integer"
            ) from error
        if timeout <= 0:
            raise ValueError(
                "EXMEKH_LIB_TOKENIZER_DOWNLOAD_TIMEOUT must be a positive integer"
            )
        return cls(download_timeout_seconds=timeout)
