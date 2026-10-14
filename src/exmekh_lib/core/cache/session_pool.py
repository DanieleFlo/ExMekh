from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from exmekh_lib.config.cache_text import CACHE_TEXT
from exmekh_lib.core.cache.session_cache import SessionCache


@dataclass(frozen=True, slots=True)
class SessionKey:
    process_id: int
    identity: str
    fingerprint: str
    directory: Path


class SessionPool:
    """Thread-safe pool managing active runtime session lifecycles."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._sessions: dict[SessionKey, SessionCache] = {}

    def get_or_create(
        self, identity: str, fingerprint: str, directory: Path
    ) -> SessionCache:
        self.validate(identity, fingerprint, directory)
        key = self.make_key(identity, fingerprint, directory)
        return self.create_session(key, directory)

    def has(self, identity: str, fingerprint: str, directory: Path) -> bool:
        """Check if an active session exists in the pool."""
        if not isinstance(identity, str) or not identity.strip():
            return False
        if not isinstance(fingerprint, str) or not isinstance(directory, Path):
            return False
        key = self.make_key(identity, fingerprint, directory)
        return self.find_active_session(key) is not None

    def close_all(self) -> None:
        """Close all active sessions and clear the pool."""
        with self._lock:
            for session in list(self._sessions.values()):
                session.close()
            self._sessions.clear()

    @property
    def sessions(self) -> dict[SessionKey, SessionCache]:
        """Return a snapshot; callers cannot mutate the pool."""
        with self._lock:
            return self._sessions.copy()

    def find_active_session(self, key: SessionKey) -> SessionCache | None:
        """Find an existing active session, ignoring closed ones."""
        with self._lock:
            session = self._sessions.get(key)
            if session is not None:
                if not session.is_closed:
                    return session
                del self._sessions[key]
            return None

    def create_session(self, key: SessionKey, directory: Path) -> SessionCache:
        """Instantiate a new session and register it in the pool."""
        self.validate(key.identity, key.fingerprint, directory)
        if key != self.make_key(key.identity, key.fingerprint, directory):
            raise ValueError(CACHE_TEXT.session_key_does_not_match_directory_or_process)
        with self._lock:
            existing = self._sessions.get(key)
            if existing is not None and not existing.is_closed:
                return existing
            session = SessionCache(directory.resolve())
            self._sessions[key] = session
            return session

    @staticmethod
    def make_key(identity: str, fingerprint: str, directory: Path) -> SessionKey:
        """Generate process-isolated session key."""
        return SessionKey(os.getpid(), identity, fingerprint, directory.resolve())

    @staticmethod
    def validate(identity: str, fingerprint: str, directory: Path) -> None:
        """Validate input arguments for session retrieval."""
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError(CACHE_TEXT.identity_must_be_a_non_empty_string)
        if not isinstance(fingerprint, str):
            raise TypeError(CACHE_TEXT.fingerprint_must_be_a_string)
        if not isinstance(directory, Path):
            raise TypeError(CACHE_TEXT.directory_must_be_a_path)
