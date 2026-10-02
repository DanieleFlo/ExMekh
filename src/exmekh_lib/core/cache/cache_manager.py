from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, ClassVar, Final, Literal

from exmekh_lib.config.cache import CACHE_FORMAT
from exmekh_lib.config.storage import DEFAULT_STORAGE_SETTINGS, StorageSettings
from exmekh_lib.config.tokenizer import QWEN_TOKENIZER
from exmekh_lib.core.cache.node_cache import NodeCache
from exmekh_lib.core.cache.session_cache import SessionCache
from exmekh_lib.core.cache.session_pool import SessionPool
from exmekh_lib.core.cache.tool_call_record import CalledToolMetadata, ToolCallRecord
from exmekh_lib.core.models.assistant_message import ToolCall


class CacheManager:
    """Central cache manager coordinating runtime sessions and node tool calling.

    Unifies common project caching logic:
      1. Deterministic SHA-256 namespace hashing.
      2. Root directory and temporary session path resolution.
      3. Process-level session sharing and synchronization via SessionPool.
      4. Node-scoped tool call caching within the unified session.
    """

    DEFAULT_TOKENIZER_IDENTITY: Final[str] = QWEN_TOKENIZER.identity
    DEFAULT_ROOT_DIR: Final[Path] = DEFAULT_STORAGE_SETTINGS.tokenizer_cache_directory
    DEFAULT_FINGERPRINT: Final[str] = CACHE_FORMAT.default_fingerprint

    _pool: ClassVar[SessionPool] = SessionPool()

    @staticmethod
    def compute_namespace(identity: str) -> str:
        """Deterministically calculate the SHA-256 namespace from an identity string."""
        if not isinstance(identity, str):
            raise TypeError("identity must be a string")
        if not identity.strip():
            raise ValueError("identity must be a non-empty string")
        return sha256(identity.encode("utf-8")).hexdigest()

    @classmethod
    def resolve_root(cls, cache_directory: Path | None = None) -> Path:
        """Resolve the root storage directory for caching."""
        if cache_directory is not None and not isinstance(cache_directory, Path):
            raise TypeError("cache_directory must be a Path or None")
        return (
            cache_directory
            or StorageSettings.from_environment().tokenizer_cache_directory
        ).resolve()

    @classmethod
    def resolve_temp_directory(
        cls,
        identity: str,
        cache_directory: Path | None = None,
    ) -> Path:
        """Resolve the temporary directory path for a specific identity."""
        root = cls.resolve_root(cache_directory)
        namespace = cls.compute_namespace(identity)
        return root / namespace / CACHE_FORMAT.temporary_folder

    @classmethod
    def get_session(
        cls,
        tokenizer_identity: str,
        vocabulary_fingerprint: str,
        directory: Path,
    ) -> SessionCache:
        """Retrieve or create the shared session for the current process, identity, and fingerprint."""
        return cls._pool.get_or_create(
            tokenizer_identity, vocabulary_fingerprint, directory
        )

    @classmethod
    def get_node_cache(
        cls,
        node_id: str,
        *,
        tokenizer_identity: str | None = None,
        vocabulary_fingerprint: str | None = None,
        cache_directory: Path | None = None,
    ) -> NodeCache:
        """Access the NodeCache for a specific node in the active session."""
        identity = (
            cls.DEFAULT_TOKENIZER_IDENTITY
            if tokenizer_identity is None
            else tokenizer_identity
        )
        fingerprint = (
            cls.DEFAULT_FINGERPRINT
            if vocabulary_fingerprint is None
            else vocabulary_fingerprint
        )
        temp_dir = cls.resolve_temp_directory(identity, cache_directory)
        session = cls.get_session(identity, fingerprint, temp_dir)
        return session.node(node_id)

    @classmethod
    def store_tool_call(
        cls,
        node_id: str,
        record: ToolCallRecord,
        *,
        tokenizer_identity: str | None = None,
        vocabulary_fingerprint: str | None = None,
        cache_directory: Path | None = None,
    ) -> Path:
        """Persist a tool call for the given node in the active session."""
        node_cache = cls.get_node_cache(
            node_id,
            tokenizer_identity=tokenizer_identity,
            vocabulary_fingerprint=vocabulary_fingerprint,
            cache_directory=cache_directory,
        )
        return node_cache.store(record)

    @classmethod
    def record_call(
        cls,
        node_id: str,
        tool_call: CalledToolMetadata | ToolCall,
        result: Any,
        id: str | None = None,
        *,
        skill_path: str = "/",
        thinking: str | None = None,
        source: Literal["Tool", "DeepSkill"] = "Tool",
        tokenizer_identity: str | None = None,
        vocabulary_fingerprint: str | None = None,
        cache_directory: Path | None = None,
    ) -> ToolCallRecord:
        """Record and persist a tool call generically for the given node in the active session."""
        node_cache = cls.get_node_cache(
            node_id,
            tokenizer_identity=tokenizer_identity,
            vocabulary_fingerprint=vocabulary_fingerprint,
            cache_directory=cache_directory,
        )
        return node_cache.record_call(
            tool_call,
            result=result,
            id=id,
            skill_path=skill_path,
            thinking=thinking,
            source=source,
        )

    @classmethod
    def get_tool_call(
        cls,
        node_id: str,
        call_id: str,
        *,
        tokenizer_identity: str | None = None,
        vocabulary_fingerprint: str | None = None,
        cache_directory: Path | None = None,
    ) -> ToolCallRecord | None:
        """Retrieve a tool call record for the given node and ID."""
        node_cache = cls.get_node_cache(
            node_id,
            tokenizer_identity=tokenizer_identity,
            vocabulary_fingerprint=vocabulary_fingerprint,
            cache_directory=cache_directory,
        )
        return node_cache.get(call_id)

    @classmethod
    def get_node_tool_calls(
        cls,
        node_id: str,
        *,
        tokenizer_identity: str | None = None,
        vocabulary_fingerprint: str | None = None,
        cache_directory: Path | None = None,
    ) -> list[ToolCallRecord]:
        """Retrieve all tool call records for the given node in the active session."""
        node_cache = cls.get_node_cache(
            node_id,
            tokenizer_identity=tokenizer_identity,
            vocabulary_fingerprint=vocabulary_fingerprint,
            cache_directory=cache_directory,
        )
        return node_cache.get_all()

    @classmethod
    def close_all(cls) -> None:
        """Close all active sessions and delete all temporary files."""
        cls._pool.close_all()

    @classmethod
    def reset(cls) -> None:
        """Reset the cache manager, closing all sessions."""
        cls.close_all()
