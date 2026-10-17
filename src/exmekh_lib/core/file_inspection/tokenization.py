"""Bind model tokenization to file tools in their execution thread."""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

from exmekh_lib.config.file_inspection import FILE_INSPECTION_TEXT
from exmekh_lib.core.tokenizers.text_budget import TextTokenizationError
from exmekh_lib.interfaces.tokenizer import Tokenizer

_tokenizer: ContextVar[Tokenizer | None] = ContextVar(
    "inspection_tokenizer", default=None
)


@contextmanager
def inspection_tokenizer_context(tokenizer: Tokenizer) -> Generator[None]:
    token = _tokenizer.set(tokenizer)
    try:
        yield
    finally:
        _tokenizer.reset(token)


def current_inspection_tokenizer() -> Tokenizer:
    tokenizer = _tokenizer.get()
    if tokenizer is None:
        raise TextTokenizationError(FILE_INSPECTION_TEXT.tokenizer_required)
    return tokenizer
