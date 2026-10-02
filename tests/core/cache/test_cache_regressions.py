from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

from exmekh_lib.capabilities.tools.search_tool_outputs.tool_call_search import (
    ToolCallSearch,
)
from exmekh_lib.core.cache import SessionCache, SessionPool, ToolCallRecord
from exmekh_lib.core.cache.paths import cache_component
from exmekh_lib.core.cache.tool_call_store import ToolCallStore
from exmekh_lib.core.models.assistant_message import FunctionCall, ToolCall
from exmekh_lib.core.models.tool_output_reference import ToolOutputReference
from exmekh_lib.core.tool_calls import (
    ToolCallData,
    ToolCallReader,
    ToolCallSaver,
)
from exmekh_lib.utils.file import read_json_from_path, write_json_atomically


def test_tool_call_record_rejects_non_dictionary_execution_metadata():
    with pytest.raises(TypeError, match="execution must be a dictionary or None"):
        ToolCallRecord("work", "output", {}, "result", execution=[])


@pytest.mark.parametrize("user_message", [[], "invalid", 42])
def test_cache_ignores_records_with_malformed_user_message(tmp_path, user_message):
    record = ToolCallRecord(
        "steering_message", "guidance", {}, "text", source="UserMessage"
    )
    store = ToolCallStore(tmp_path)
    path = store.save(record)
    payload = record.to_dict()
    payload["user_message"] = user_message
    write_json_atomically(path, payload)

    assert store.get(record.id) is None
    assert store.get_all() == []


@pytest.mark.parametrize(
    "references", [(), (ToolOutputReference("work", "tool-output"),)]
)
def test_skill_output_references_round_trip_without_changing_the_result(
    tmp_path, references
):
    result = {"text": "done", "items": []}
    record = ToolCallRecord(
        "skill", "skill-output", {}, result, source="DeepSkill", used_tools=references
    )
    store = ToolCallStore(tmp_path)

    payload = read_json_from_path(store.save(record))

    assert payload["result"] == result
    assert payload["used_tools"] == [reference.to_dict() for reference in references]
    assert store.get(record.id) == record


@pytest.mark.parametrize("references", [[], ({"name": "work", "id": "output"},)])
def test_cache_records_reject_mutable_or_untyped_tool_references(references):
    with pytest.raises(TypeError, match="tuple of ToolOutputReference"):
        ToolCallRecord("skill", "output", {}, "done", used_tools=references)


def test_cache_records_reject_malformed_serialized_tool_references():
    data = {
        "name": "skill",
        "id": "output",
        "args": {},
        "result": "done",
        "used_tools": {},
    }

    with pytest.raises(TypeError, match="Serialized used_tools must be a list"):
        ToolCallRecord.from_dict(data)


def test_sessions_are_scoped_to_resolved_directory(tmp_path):
    pool = SessionPool()
    try:
        first = pool.get_or_create("model", "v1", tmp_path / "first")
        second = pool.get_or_create("model", "v1", tmp_path / "second")
        assert first is not second
        assert pool.get_or_create("model", "v1", tmp_path / "first" / ".") is first
        snapshot = pool.sessions
        snapshot.clear()
        assert pool.has("model", "v1", tmp_path / "first")
    finally:
        pool.close_all()


def test_session_identity_is_not_lossily_normalized(tmp_path):
    pool = SessionPool()
    try:
        assert pool.get_or_create("model", "v1", tmp_path) is not pool.get_or_create(
            " model ", "v1", tmp_path
        )
    finally:
        pool.close_all()


def test_concurrent_session_creation_preserves_directory_isolation(tmp_path):
    pool = SessionPool()
    try:
        with ThreadPoolExecutor(max_workers=8) as workers:
            sessions = list(
                workers.map(
                    lambda i: pool.get_or_create("model", "v1", tmp_path / str(i % 2)),
                    range(40),
                )
            )
        assert len({id(session) for session in sessions}) == 2
        assert all(
            session is sessions[index % 2] for index, session in enumerate(sessions)
        )
    finally:
        pool.close_all()


@pytest.mark.parametrize(
    "identifier",
    [
        "a/b",
        "a\\b",
        "..",
        "../outside",
        "C:\\outside",
        "CON",
        "nul",
        "Call",
        "é",
        "x" * 500,
    ],
)
def test_ids_are_opaque_and_portable(tmp_path, identifier):
    with SessionCache(tmp_path) as session:
        node = session.node(identifier)
        record = node.record("tool", identifier, {}, "value")
        assert node.get(identifier) == record
        assert node.list_ids() == [identifier]
        files = list(session.session_dir.rglob("*.json"))
        assert len(files) == 1
        assert files[0].is_relative_to(session.session_dir)
        assert len(files[0].relative_to(session.session_dir).parts) == 3


def test_colliding_ids_remain_distinct_for_nodes_and_calls(tmp_path):
    identifiers = ["a/b", "a\\b", "a_b", "Call", "call", "node.", "node"]
    with SessionCache(tmp_path) as session:
        for identifier in identifiers:
            session.node(identifier).record("tool", "same", {}, identifier)
            session.node("shared").record("tool", identifier, {}, identifier)
        for identifier in identifiers:
            assert session.node(identifier).get("same").result == identifier
            assert session.node("shared").get(identifier).result == identifier
        assert len(session.node("shared").get_all()) == len(identifiers)


def test_state_none_deletion_and_late_node_view(tmp_path):
    with SessionCache(tmp_path) as session:
        session.set_node_state("node", "nullable", None)
        node = session.node(" node ")
        assert node.has_state("nullable")
        assert node.delete_state("nullable") is True
        assert node.delete_state("nullable") is False


def test_read_operations_do_not_create_storage(tmp_path):
    with SessionCache(tmp_path) as session:
        node = session.node("absent")
        assert node.get("missing") is None
        assert node.get_all() == []
        assert node.list_ids() == []
        assert not (session.session_dir / "tool_calls").exists()


def test_record_snapshots_mutable_inputs_and_supports_cycles(tmp_path):
    arguments = {"items": [1]}
    cycle = []
    cycle.append(cycle)
    output = {"path": Path("example"), "cycle": cycle, "nan": float("nan")}
    record = ToolCallRecord("tool", "id", arguments, output, "/skill")
    arguments["items"].append(2)
    output["path"] = "changed"
    store = ToolCallStore(tmp_path)
    path = store.save(record)
    payload = read_json_from_path(path)
    assert payload["args"] == {"items": [1]}
    assert payload["result"]["cycle"] == ["<circular reference>"]
    assert payload["result"]["nan"] == "nan"
    assert payload["result"]["path"] != "changed"
    assert store.get("id") == record


def test_shared_references_are_not_cycles():
    shared = [1]
    record = ToolCallRecord("tool", "id", {}, [shared, shared])
    assert record.result == [[1], [1]]


def test_failed_save_preserves_existing_record_and_cleans_temp_files(tmp_path):
    store = ToolCallStore(tmp_path)
    original = ToolCallRecord("tool", "id", {}, "original")
    store.save(original)
    with (
        patch.object(Path, "replace", side_effect=OSError("disk failure")),
        pytest.raises(OSError, match="disk failure"),
    ):
        store.save(ToolCallRecord("tool", "id", {}, "replacement"))
    assert store.get("id") == original
    assert not list(tmp_path.glob("*.tmp"))


def test_reader_and_session_share_one_record_format(tmp_path):
    with SessionCache(tmp_path) as session:
        node = session.node("node")
        record = node.record("tool", "id", {}, {"value": 1}, skill_path="/parent/child")
        reader = ToolCallReader(directory=session.session_dir / "tool_calls" / "node")
        assert reader.directory == session.session_dir / "tool_calls" / "node"
        assert reader.get("id") == record
        assert ToolCallData is ToolCallRecord


def test_old_records_default_to_root_skill_path():
    record = ToolCallRecord.from_dict(
        {"name": "tool", "id": "id", "args": {}, "result": None}
    )
    assert record.skill_path == "/"
    assert record.source == "Tool"


def test_record_rejects_invalid_source():
    with pytest.raises(ValueError, match="source must be"):
        ToolCallRecord("tool", "id", {}, None, source="ExitTool")  # type: ignore[arg-type]


@pytest.mark.parametrize("path", [None, 3, "relative", ""])
def test_record_rejects_invalid_skill_path(path):
    with pytest.raises((TypeError, ValueError)):
        ToolCallRecord("tool", "id", {}, None, path)


def test_malformed_or_mismatched_files_are_not_returned(tmp_path):
    store = ToolCallStore(tmp_path)
    (tmp_path / "broken.json").write_text("not json")
    (tmp_path / "list.json").write_text("[]")
    (tmp_path / "wrong.json").write_text(
        '{"name":"tool","id":"other","args":{},"result":1}'
    )
    assert store.get("broken") is None
    assert store.get("wrong") is None
    assert "broken" not in store.list_ids()


@pytest.mark.parametrize("folder", ["../escape", "C:\\escape", "..", "CON", "a/b"])
def test_standalone_archive_rejects_path_in_folder_name(tmp_path, folder):
    with pytest.raises(ValueError, match="portable"):
        ToolCallSaver(
            tokenizer_identity="model", cache_directory=tmp_path, folder_name=folder
        )


def test_call_adapters_preserve_arguments_and_allow_inferred_ids(tmp_path):
    call = ToolCall("call", FunctionCall("tool", {"value": 3}))
    saver = ToolCallSaver(directory=tmp_path)
    record = saver.record_call(call, "result", skill_path="/skill")
    assert record.id == "call"
    assert record.args == {"value": 3}
    assert record.skill_path == "/skill"


def test_search_tool_is_scoped_and_supports_path_query_and_pagination(tmp_path):
    with SessionCache(tmp_path) as session:
        node = session.node("local")
        node.record("fetch", "one", {}, "HELLO", skill_path="/skill")
        node.record("fetch", "two", {}, "hello again", skill_path="/skill")
        node.record("fetch", "three", {}, "hello", skill_path="/other")
        session.node("other").record(
            "fetch", "secret", {}, "hello", skill_path="/skill"
        )
        tool = ToolCallSearch(node).as_tool()
        page = tool.execute(query="hello", skill_path="/skill", limit=1)
        assert page["total"] == 2
        assert page["next_offset"] == 1
        second = tool.execute(query="hello", skill_path="/skill", limit=1, offset=1)
        assert second["next_offset"] is None
        assert page["records"][0]["id"] != second["records"][0]["id"]
        assert all(item["id"] != "secret" for item in tool.execute()["records"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"offset": -1},
        {"limit": 0},
        {"limit": True},
        {"limit": 101},
        {"query": None},
        {"tool_name": ""},
        {"skill_path": "relative"},
    ],
)
def test_search_rejects_invalid_parameters(tmp_path, kwargs):
    with SessionCache(tmp_path) as session, pytest.raises((TypeError, ValueError)):
        ToolCallSearch(session.node("node")).search(**kwargs)


def test_legacy_filename_is_read_without_reintroducing_id_collisions(tmp_path):
    old_record = ToolCallRecord("tool", "a/b", {}, "old")
    write_json_atomically(tmp_path / "a_b.json", old_record.to_dict())
    store = ToolCallStore(tmp_path)
    assert store.get("a/b") == old_record
    assert store.get("a_b") is None
    new_record = ToolCallRecord("tool", "a/b", {}, "new")
    store.save(new_record)
    assert store.get("a/b") == new_record
    assert store.get_all() == [new_record]


def test_search_rejects_a_closed_session(tmp_path):
    session = SessionCache(tmp_path)
    search = ToolCallSearch(session.node("node"))
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        search.search()


@pytest.mark.parametrize("capacity", [True, "10", 1.5, 0, -1])
def test_counter_rejects_invalid_capacity(tmp_path, capacity):
    with SessionCache(tmp_path) as session:
        with pytest.raises((TypeError, ValueError)):
            session.reserve_next_index(capacity)
        assert session.reserve_next_index(1) == 0


@pytest.mark.parametrize(
    "thinking", [None, "", "  I am reasoning about the tool.\nSecond line: city.  "]
)
def test_thinking_round_trip_across_all_cache_entry_points(tmp_path, thinking):

    call = ToolCall("call", FunctionCall("tool", {"value": 3}))

    saver = ToolCallSaver(directory=tmp_path / "archive")

    from_call = saver.record_call(call, "result", thinking=thinking)

    assert from_call.thinking == thinking

    assert ToolCallReader(directory=saver.directory).get("call").thinking == thinking

    direct = saver.record("tool", "direct", {}, None, thinking=thinking)

    assert direct.thinking == thinking

    with SessionCache(tmp_path / "sessions") as session:
        cache = session.node("node")

        assert cache.record_call(call, "result", thinking=thinking).thinking == thinking

        assert (
            cache.record("tool", "direct", {}, None, thinking=thinking).thinking
            == thinking
        )

        assert cache.get("call").thinking == thinking


def test_legacy_records_have_no_thinking():

    record = ToolCallRecord.from_dict(
        {"name": "tool", "id": "call", "args": {}, "result": 3}
    )

    assert record.thinking is None


@pytest.mark.parametrize("thinking", [1, [], {}, True])
def test_record_rejects_non_text_thinking(thinking):

    with pytest.raises(TypeError, match="thinking must be a string or None"):
        ToolCallRecord("tool", "call", {}, None, thinking=thinking)


@pytest.mark.parametrize("call_id", ["", "  ", 1, False])
def test_record_rejects_invalid_provider_call_id(call_id):
    with pytest.raises(ValueError, match="tool_call_id"):
        ToolCallRecord("tool", "call", {}, None, tool_call_id=call_id)


@pytest.mark.parametrize("identifier", [None, "", "  ", 1])
def test_invalid_identifiers_cannot_create_cache_storage(tmp_path, identifier):
    with pytest.raises(ValueError):
        cache_component(identifier)
    with SessionCache(tmp_path) as session:
        with pytest.raises(ValueError, match="node_id"):
            session.node(identifier)
        assert not list(session.session_dir.iterdir())
    store = ToolCallStore(tmp_path / "archive")
    assert store.get(identifier) is None
    assert not store.directory.exists()


def test_store_rejects_invalid_record_before_creating_directory(tmp_path):
    store = ToolCallStore(tmp_path / "archive")
    with pytest.raises(TypeError, match="ToolCallRecord"):
        store.save({"id": "call"})
    assert not store.directory.exists()


def test_session_rejects_non_path_directory():
    with pytest.raises(TypeError, match="directory"):
        SessionCache("cache")


def test_pool_rejects_key_for_another_directory(tmp_path):
    pool = SessionPool()
    key = pool.make_key("model", "v1", tmp_path / "first")
    with pytest.raises(ValueError, match="session key"):
        pool.create_session(key, tmp_path / "second")
    assert pool.sessions == {}
    assert not (tmp_path / "second").exists()


def test_session_refuses_redirected_node_directory(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    with SessionCache(tmp_path / "sessions") as session:
        node_directory = session.session_dir / "tool_calls" / "node"
        node_directory.parent.mkdir()
        node_directory.symlink_to(outside, target_is_directory=True)
        try:
            with pytest.raises(ValueError, match="redirect"):
                session.node("node").record("tool", "call", {}, "value")
            assert list(outside.iterdir()) == []
        finally:
            node_directory.unlink()


def test_store_does_not_follow_record_symlinks(tmp_path):
    outside = ToolCallStore(tmp_path / "outside")
    record = ToolCallRecord("tool", "call", {}, "private")
    original = outside.save(record)
    archive = tmp_path / "archive"
    archive.mkdir()
    link = archive / original.name
    link.symlink_to(original)
    try:
        store = ToolCallStore(archive)
        assert store.get("call") is None
        assert store.get_all() == []
        assert outside.get("call") == record
    finally:
        link.unlink()


@pytest.mark.parametrize("canonical_first", [False, True])
def test_canonical_record_wins_regardless_of_directory_order(tmp_path, canonical_first):
    legacy = ToolCallRecord("tool", "a/b", {}, "old")
    legacy_path = tmp_path / "a_b.json"
    write_json_atomically(legacy_path, legacy.to_dict())
    store = ToolCallStore(tmp_path)
    current = ToolCallRecord("tool", "a/b", {}, "new")
    canonical_path = store.save(current)
    paths = (
        [canonical_path, legacy_path]
        if canonical_first
        else [legacy_path, canonical_path]
    )
    with patch.object(Path, "glob", return_value=iter(paths)):
        assert store.get_all() == [current]
