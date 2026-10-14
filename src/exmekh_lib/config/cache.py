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
    output_cleanup_lock: str = ".cleanup.lock"
    output_lease_filename: str = ".lease"
    output_text_suffix: str = ".txt"
    output_pending_prefix: str = ".pending-"
    output_part_digits: int = 4
    output_text_encoding: str = "utf-8"
    output_text_errors: str = "surrogatepass"
    output_digest_separator: bytes = b"\0"
    output_lease_mode: str = "r+b"
    output_lease_permissions: int = 0o600
    output_lease_seed: bytes = b"0"
    output_lease_length: int = 1
    output_access_context_name: str = "tool_output_file_access"


CACHE_FORMAT = CacheFormat()
