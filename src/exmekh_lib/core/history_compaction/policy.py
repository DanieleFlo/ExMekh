"""Map importance to retention; summary length is governed only by prompts."""

from math import ceil

from exmekh_lib.config.history_compaction import HISTORY_COMPACTION


def normalize_scores(scores: dict[str, int]) -> dict[str, int]:
    """Lower scores by one offset, preserving every tie at the retention cutoff."""
    threshold = HISTORY_COMPACTION.summary_score_threshold
    eligible = sorted(
        (score for score in scores.values() if score >= threshold), reverse=True
    )
    limit = max(
        HISTORY_COMPACTION.minimum_summary_candidates,
        ceil(len(scores) * HISTORY_COMPACTION.summary_fraction),
    )
    if len(eligible) <= limit:
        return dict(scores)
    cutoff = eligible[limit - 1]
    if eligible[-1] == cutoff:
        return dict(scores)
    offset = cutoff - threshold
    return {entry: max(1, score - offset) for entry, score in scores.items()}


def summary_prompt(score: int, token_count: int) -> str | None:
    """None keeps the original; an empty prompt keeps only the reference."""
    if score < HISTORY_COMPACTION.summary_score_threshold:
        return ""
    if score < 50:
        return HISTORY_COMPACTION.sentence_prompt
    if score < 80:
        return None if token_count < 100 else HISTORY_COMPACTION.short_prompt
    if score < 95:
        return None if token_count < 200 else HISTORY_COMPACTION.detailed_prompt
    return None if token_count < 5000 else HISTORY_COMPACTION.essential_prompt
