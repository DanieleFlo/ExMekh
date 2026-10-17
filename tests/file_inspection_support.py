"""Deterministic tokenization for standalone file inspection fixtures."""

from functools import partial

from exmekh_lib.core.file_inspection.service import (
    inspect_file as inspect_with_tokenizer,
)
from tests.tokenizer_support import StubTokenizer

inspect_file = partial(inspect_with_tokenizer, tokenizer=StubTokenizer())
