from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CacheFormat:
    """Fixed names shared by tokenizer, session, and tool-call storage."""

    default_fingerprint: str = "default"
    temporary_folder: str = "temp"
    vocabulary_filename: str = "vocabulary.json"
    counter_filename: str = "counter.json"
    session_prefix: str = "session-"
    tool_calls_folder: str = "tool_calls"
    record_suffix: str = ".json"


CACHE_FORMAT = CacheFormat()
