from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from urllib.request import urlopen

from qs.core.tokenizers.sequential_tokenizer import SequentialTokenizer


class QwenTokenizer(SequentialTokenizer):
    MODEL = "Qwen/Qwen3.8-27B"

    def __init__(self, cache_directory: Path | None = None) -> None:
        super().__init__(f"{self.MODEL}:main:ascii-v1", cache_directory)

    def _load_vocabulary_tokens(self) -> Iterable[str]:
        url = f"https://huggingface.co/{self.MODEL}/resolve/main/tokenizer.json"
        with urlopen(url, timeout=60) as response:
            data = json.load(response)
        vocabulary = data["model"]["vocab"]
        special_tokens = {
            token["content"]
            for token in data.get("added_tokens", [])
            if token.get("special", False)
        }
        return (token for token in vocabulary if token not in special_tokens)
