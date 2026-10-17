"""Contract and failure boundaries for model-token text budgets."""

import pytest

from exmekh_lib.core.tokenizers.text_budget import (
    TextTokenizationError,
    TokenTextBudget,
)
from tests.tokenizer_support import PairTokenizer, StubTokenizer


@pytest.mark.parametrize("tokenizer_type", [StubTokenizer, PairTokenizer])
@pytest.mark.parametrize("limit", [0, 1, 2, 5])
def test_text_and_suffix_together_fit_the_budget(tokenizer_type, limit):
    tokenizer = tokenizer_type()
    budget = TokenTextBudget(tokenizer, limit)
    result = budget.truncate("café 😀 漢字 repeated", suffix="…")
    assert tokenizer.count_tokens(result) <= limit


def test_untruncated_text_keeps_whitespace_and_does_not_gain_a_suffix():
    text = "  café\r\n"
    assert TokenTextBudget(PairTokenizer(), 20).truncate(text, suffix="…") == text


@pytest.mark.parametrize("limit", [-1, True, 1.5, "5"])
def test_invalid_configuration_is_rejected_before_tokenization(limit):
    with pytest.raises(ValueError, match="nonnegative integer"):
        TokenTextBudget(StubTokenizer(), limit)


@pytest.mark.parametrize("count", [-1, True, None, 1.5, "3"])
def test_invalid_token_counts_are_reported_as_tokenizer_failures(count):
    class InvalidTokenizer(StubTokenizer):
        def count_tokens(self, text):
            return count

    with pytest.raises(TextTokenizationError, match="invalid token count"):
        TokenTextBudget(InvalidTokenizer(), 2).truncate("abcdef")


@pytest.mark.parametrize("prefix", [None, 123, "different text", "abcdef"])
def test_invalid_or_oversized_prefixes_cannot_escape_the_budget(prefix):
    class InvalidTokenizer(StubTokenizer):
        def truncate_text(self, text, start_tokens, end_tokens=0):
            return prefix

    with pytest.raises(TextTokenizationError, match="unchanged prefix"):
        TokenTextBudget(InvalidTokenizer(), 2).truncate("abcdef")


def test_unstable_tokenizer_cannot_start_an_unbounded_retry_loop():
    class UnstableTokenizer(StubTokenizer):
        empty_calls = 0

        def count_tokens(self, text):
            if text:
                return len(text)
            self.empty_calls += 1
            return self.empty_calls - 1

    with pytest.raises(TextTokenizationError, match="unchanged prefix"):
        TokenTextBudget(UnstableTokenizer(), 0).truncate("abcdef", suffix="…")
