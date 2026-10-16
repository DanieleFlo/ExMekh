"""Map importance to retention; summary length is governed only by prompts."""

from exmekh_lib.config.history_compaction import HISTORY_COMPACTION


def summary_prompt(score: int, token_count: int) -> str | None:
    """None keeps the original; an empty prompt keeps only the reference."""
    if score < 10:
        return ""
    if score < 50:
        return HISTORY_COMPACTION.sentence_prompt
    if score < 80:
        return None if token_count < 100 else HISTORY_COMPACTION.short_prompt
    if score < 95:
        return None if token_count < 200 else HISTORY_COMPACTION.detailed_prompt
    return None if token_count < 5000 else HISTORY_COMPACTION.essential_prompt
