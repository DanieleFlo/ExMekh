"""Standalone tokenizer doubles shared by unit and integration tests."""

from exmekh_lib.config.storage import DEFAULT_STORAGE_SETTINGS
from exmekh_lib.core.cache.cache_manager import CacheManager
from exmekh_lib.core.cache.session_cache import SessionCache
from exmekh_lib.interfaces.tokenizer import Tokenizer


class StubTokenizer(Tokenizer):
    """Use one Unicode code point per token without loading a model."""

    def count_tokens(self, text: str) -> int:
        return len(text)

    def truncate_text(self, text: str, start_tokens: int, end_tokens: int = 0) -> str:
        if self.count_tokens(text) <= start_tokens + end_tokens:
            return text
        return text[:start_tokens] + (text[-end_tokens:] if end_tokens else "")

    def initialize(self) -> None:
        pass

    def generate_id(self) -> str:
        raise RuntimeError("No tokenizer session for stub")

    def get_session(self) -> SessionCache:
        return CacheManager.get_session(
            "stub-node", "default", DEFAULT_STORAGE_SETTINGS.tokenizer_cache_directory
        )


class PairTokenizer(StubTokenizer):
    """Treat each pair of Unicode code points as one token."""

    def count_tokens(self, text: str) -> int:
        return (len(text) + 1) // 2

    def truncate_text(self, text: str, start_tokens: int, end_tokens: int = 0) -> str:
        if self.count_tokens(text) <= start_tokens + end_tokens:
            return text
        return text[: 2 * start_tokens] + (
            text[-2 * end_tokens :] if end_tokens else ""
        )
