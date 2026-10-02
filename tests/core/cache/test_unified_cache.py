from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from exmekh_lib.core.cache import (
    CacheManager,
    NodeCache,
    SessionCache,
    SessionPool,
    ToolCallRecord,
)
from exmekh_lib.core.models.assistant_message import FunctionCall, ToolCall
from exmekh_lib.core.tokenizers.sequential_tokenizer import SequentialTokenizer
from exmekh_lib.core.tokenizers.tokenizer_session import TokenizerSession
from exmekh_lib.core.tokenizers.tokenizer_session_registry import (
    TokenizerSessionRegistry,
)
from exmekh_lib.utils.file import read_json_from_path


@pytest.fixture(autouse=True)
def clean_sessions():
    CacheManager.reset()
    yield
    CacheManager.reset()


class MockTokenizer(SequentialTokenizer):
    def count_tokens(self, text: str) -> int:
        return len(text)

    def truncate_text(self, text: str, start_tokens: int, end_tokens: int = 0) -> str:
        return text[:start_tokens] + (text[-end_tokens:] if end_tokens else "")

    def __init__(self, directory: Path, identity: str = "mock-vendor/model:v1"):
        super().__init__(identity, directory)

    def _load_vocabulary_tokens(self):
        return ["alpha", "beta", "gamma", "delta"]


class DummyTool:
    def __init__(self, name: str, call_id: str, arguments: dict):
        self.name = name
        self.call_id = call_id
        self.arguments = arguments
        self.tool_call = ToolCall(
            id=call_id, function=FunctionCall(name=name, arguments=arguments)
        )


@pytest.mark.parametrize("identity,error", [(None, TypeError), ("  ", ValueError)])
def test_namespace_rejects_invalid_identity(identity, error):
    with pytest.raises(error):
        CacheManager.compute_namespace(identity)


def test_cache_root_requires_path():
    with pytest.raises(TypeError, match="cache_directory"):
        CacheManager.resolve_root("cache")


def test_manager_adapters_preserve_scope_thinking_and_node_isolation(tmp_path):
    options = {
        "tokenizer_identity": "model",
        "vocabulary_fingerprint": "v1",
        "cache_directory": tmp_path,
    }
    record = ToolCallRecord("tool", "saved", {}, "value", "/child", thinking="reason")
    path = CacheManager.store_tool_call("node", record, **options)
    assert path.is_file()
    assert CacheManager.get_tool_call("node", "saved", **options) == record
    call = ToolCall("provider-call", FunctionCall("tool", {"value": 3}))
    recorded = CacheManager.record_call(
        "node",
        call,
        6,
        id="recorded",
        skill_path="/child",
        thinking="thought",
        **options,
    )
    assert recorded.tool_call_id == "provider-call"
    assert recorded.args == {"value": 3}
    assert recorded.result == 6
    assert recorded.skill_path == "/child"
    assert recorded.thinking == "thought"
    assert CacheManager.get_node_tool_calls("node", **options) == [recorded, record]
    assert CacheManager.get_node_tool_calls("other", **options) == []
    assert CacheManager.get_node_cache("node", **options).node_id == "node"


def test_get_session_initializes_tokenizer_and_registry_reset_invalidates_it(tmp_path):
    tokenizer = MockTokenizer(tmp_path)
    session = tokenizer.get_session()
    assert tokenizer.get_session() is session
    assert tokenizer.generate_id() == "alpha_alpha_alpha"
    TokenizerSessionRegistry.reset()
    assert session.is_closed
    assert not session.session_dir.exists()
    assert tokenizer.get_session() is not session
    assert tokenizer.generate_id() == "alpha_alpha_alpha"


# --- ToolCallRecord Tests ---


def test_tool_call_record_valid():
    record = ToolCallRecord(
        name="calculator",
        id="call-123",
        args={"op": "add", "values": [1, 2]},
        result=3,
    )
    assert record.name == "calculator"
    assert record.id == "call-123"
    assert record.args == {"op": "add", "values": [1, 2]}
    assert record.result == 3

    # Immutability
    with pytest.raises((AttributeError, TypeError)):
        record.name = "new_name"  # type: ignore

    # Serialization roundtrip
    d = record.to_dict()
    assert ToolCallRecord.from_dict(d) == record


def test_tool_call_record_invalid():
    with pytest.raises(TypeError):
        ToolCallRecord(name=123, id="1", args={}, result=None)  # type: ignore

    with pytest.raises(ValueError, match="cannot be empty"):
        ToolCallRecord(name="   ", id="1", args={}, result=None)

    with pytest.raises(TypeError):
        ToolCallRecord(name="tool", id=123, args={}, result=None)  # type: ignore

    with pytest.raises(ValueError, match="cannot be empty"):
        ToolCallRecord(name="tool", id="   ", args={}, result=None)

    with pytest.raises(TypeError):
        ToolCallRecord(name="tool", id="1", args="not-dict", result=None)  # type: ignore


# --- Unified Session: Tokenizer Counter + Node Tool Calls ---


def test_unified_session_holds_counter_and_node_tool_calls(tmp_path: Path):
    session = TokenizerSession(tmp_path)
    session_dir = session.session_dir

    # 1. Reserve index for tokenizer
    idx = session.reserve_next_index(100)
    assert idx == 0

    # counter.json exists
    counter_file = session_dir / "counter.json"
    assert counter_file.is_file()
    assert read_json_from_path(counter_file) == {"next_index": 1}

    # 2. Node uses the SAME session to record a tool call
    node_cache = session.node("node_worker")
    assert isinstance(node_cache, NodeCache)
    record = node_cache.record(
        name="grep",
        id="call_000",
        args={"pattern": "abc"},
        result=["line 1", "line 2"],
    )

    # Tool call file exists inside session_dir / tool_calls / node_worker / call_000.json
    tool_call_file = session_dir / "tool_calls" / "node_worker" / "call_000.json"
    assert tool_call_file.is_file()
    assert read_json_from_path(tool_call_file)["name"] == "grep"

    # Read back through node_cache
    assert node_cache.get("call_000") == record
    assert node_cache.list_ids() == ["call_000"]
    assert node_cache.has("call_000")
    assert not node_cache.has("unknown")

    # 3. Close the session: EVERYTHING in session_dir is wiped simultaneously
    session.close()
    assert not session_dir.exists()
    assert session.is_closed

    with pytest.raises(RuntimeError, match="closed"):
        session.reserve_next_index(10)

    with pytest.raises(RuntimeError, match="closed"):
        node_cache.get("call_000")


def test_node_isolation_within_same_session(tmp_path: Path):
    session = SessionCache(tmp_path)

    node_1 = session.node("node_1")
    node_2 = session.node("node_2")

    node_1.record("fetch", "c_1", {"url": "a.com"}, "res_a")
    node_2.record("fetch", "c_2", {"url": "b.com"}, "res_b")

    # Node 1 only sees c_1
    assert node_1.get("c_1") is not None
    assert node_1.get("c_2") is None
    assert node_1.list_ids() == ["c_1"]

    # Node 2 only sees c_2
    assert node_2.get("c_2") is not None
    assert node_2.get("c_1") is None
    assert node_2.list_ids() == ["c_2"]


def test_node_state_cache(tmp_path: Path):
    session = SessionCache(tmp_path)
    node = session.node("state_node")

    node.set_state("step", 42)
    assert node.has_state("step")
    assert node.get_state("step") == 42
    assert node.get_state("non_existent", default="none") == "none"

    assert session.has_node_state("state_node", "step")

    assert node.delete_state("step") is True
    assert node.delete_state("step") is False
    assert not node.has_state("step")


# --- Tokenizer Integration via CacheManager ---


def test_sequential_tokenizer_shares_session_with_cache_manager(tmp_path: Path):
    tokenizer = MockTokenizer(tmp_path)
    # Generating an ID triggers session creation via TokenizerSessionRegistry -> CacheManager
    generated_id = tokenizer.generate_id()
    assert generated_id == "alpha_alpha_alpha"

    # Now retrieve node cache for the same tokenizer identity from CacheManager
    node = CacheManager.get_node_cache(
        "analysis_node",
        tokenizer_identity=tokenizer._identity,
        vocabulary_fingerprint=tokenizer._vocabulary_fingerprint,
        cache_directory=tmp_path,
    )

    # Record a tool call with the generated ID!
    called_tool = DummyTool(
        name="tokenize_eval", call_id=generated_id, arguments={"text": "hello"}
    )
    saved_record = node.record_call(called_tool, result="success", id=generated_id)
    assert saved_record.id == generated_id
    assert saved_record.args == {"text": "hello"}

    # The tool call is recorded in the active session
    assert node.get(generated_id) == saved_record

    # Verify both counter and tool call reside under the same session folder
    session_dir = node.session_dir

    assert (session_dir / "counter.json").is_file()
    assert (
        session_dir / "tool_calls" / "analysis_node" / f"{generated_id}.json"
    ).is_file()

    # Reset closes and wipes everything
    CacheManager.reset()
    assert not session_dir.exists()


# --- Concurrency & Thread-Safety ---


def test_concurrent_tool_calls_and_index_reservation(tmp_path: Path):
    session = SessionCache(tmp_path)

    def worker(i: int):
        idx = session.reserve_next_index(1000)
        node = session.node(f"thread_node_{i % 4}")
        node.record(
            name=f"tool_{i}", id=f"call_{idx:04d}", args={"step": i}, result=idx
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, range(80)))

    # 80 indices reserved in total
    counter_data = read_json_from_path(session.session_dir / "counter.json")
    assert counter_data["next_index"] == 80

    # 20 records per node
    for n in range(4):
        node = session.node(f"thread_node_{n}")
        assert len(node.get_all()) == 20

    session.close()


# --- SessionPool Tests ---


def test_session_pool_validation_errors(tmp_path: Path):
    with pytest.raises(ValueError, match="identity must be a non-empty string"):
        SessionPool.validate("", "fp-1", tmp_path)

    with pytest.raises(TypeError, match="fingerprint must be a string"):
        SessionPool.validate("model:v1", 123, tmp_path)  # type: ignore

    with pytest.raises(TypeError, match="directory must be a Path"):
        SessionPool.validate("model:v1", "fp-1", "not-a-path")  # type: ignore


def test_session_pool_create_and_find_active_session(tmp_path: Path):
    pool = SessionPool()
    key = pool.make_key("model:v1", "fp-1", tmp_path)
    assert pool.find_active_session(key) is None
    assert not pool.has("model:v1", "fp-1", tmp_path)

    session = pool.create_session(key, tmp_path)
    assert not session.is_closed
    assert pool.has("model:v1", "fp-1", tmp_path)
    assert pool.find_active_session(key) is session

    # Closing all sessions clears the pool
    pool.close_all()
    assert pool.find_active_session(key) is None
    assert not pool.has("model:v1", "fp-1", tmp_path)
    assert len(pool.sessions) == 0


def test_session_pool_ignores_closed_session(tmp_path: Path):
    pool = SessionPool()
    key = pool.make_key("model:v1", "fp-1", tmp_path)

    session1 = pool.create_session(key, tmp_path)
    assert not session1.is_closed
    assert pool.has("model:v1", "fp-1", tmp_path)

    # Manually close session1
    session1.close()
    assert session1.is_closed
    assert pool.find_active_session(key) is None
    assert not pool.has("model:v1", "fp-1", tmp_path)

    # Creating a fresh session for the same key
    session2 = pool.create_session(key, tmp_path)
    assert session2 is not session1
    assert not session2.is_closed
    assert pool.find_active_session(key) is session2

    pool.close_all()


def test_session_pool_has_invalid_inputs(tmp_path):
    pool = SessionPool()
    assert not pool.has("", "fp-1", tmp_path)
    assert not pool.has("   ", "fp-1", tmp_path)
    assert not pool.has(123, "fp-1", tmp_path)  # type: ignore
    assert not pool.has("model:v1", 123, tmp_path)  # type: ignore


# --- CacheManager get_session Lifecycle Tests ---


def test_cache_manager_get_session_lifecycle(tmp_path: Path):
    session1 = CacheManager.get_session("model:v1", "fp-1", tmp_path)
    assert not session1.is_closed

    # Second call reuses the active session
    session2 = CacheManager.get_session("model:v1", "fp-1", tmp_path)
    assert session2 is session1

    # Closing session causes get_session to create a fresh one
    session1.close()
    session3 = CacheManager.get_session("model:v1", "fp-1", tmp_path)
    assert session3 is not session1
    assert not session3.is_closed

    CacheManager.reset()


def test_cache_manager_concurrent_get_session(tmp_path: Path):
    def get_sess():
        return CacheManager.get_session("model:v1", "fp-1", tmp_path)

    with ThreadPoolExecutor(max_workers=10) as executor:
        sessions = list(executor.map(lambda _: get_sess(), range(30)))

    first = sessions[0]
    assert all(s is first for s in sessions)
    assert not first.is_closed

    CacheManager.reset()
