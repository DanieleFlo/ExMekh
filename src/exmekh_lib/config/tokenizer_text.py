"""Fixed diagnostic and log messages for the tokenizer domain."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenizerText:
    lossless_split_required: str = (
        "Tokenizer cannot split text losslessly within the token budget"
    )
    built_and_cached_vocabulary_identity_tokens_path: str = (
        "Built and cached vocabulary: identity=%s tokens=%d path=%s"
    )
    cache_directory_must_be_a_path_or_none: str = (
        "Cache directory must be a Path or None"
    )
    cannot_be_negative: str = "{name} cannot be negative"
    invalid_cache_metadata: str = "Invalid cache metadata"
    invalid_cached_vocabulary: str = "Invalid cached vocabulary"
    invalid_tokenizer_cache: str = "Invalid tokenizer cache: {path}"
    loaded_cached_vocabulary_identity_tokens_path: str = (
        "Loaded cached vocabulary: identity=%s tokens=%d path=%s"
    )
    max_tokens_must_be_positive: str = "max_tokens must be positive"
    must_be_an_integer: str = "{name} must be an integer"
    text_must_be_a_string: str = "text must be a string"
    tokenizer_cannot_split_a_character_within_the_token_budget: str = (
        "Tokenizer cannot split a character within the token budget"
    )
    tokenizer_did_not_cover_the_complete_source_text: str = (
        "Tokenizer did not cover the complete source text"
    )
    tokenizer_has_no_ascii_alphanumeric_tokens: str = (
        "Tokenizer has no ASCII alphanumeric tokens"
    )
    tokenizer_identity_must_be_a_string: str = "Tokenizer identity must be a string"
    tokenizer_identity_must_not_be_empty: str = "Tokenizer identity must not be empty"


TOKENIZER_TEXT = TokenizerText()
