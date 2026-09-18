from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path

import pytest

from qs.core.models.assistant_message import FunctionCall, ToolCall
from qs.core.tool_calls import ToolCallData, ToolCallReader, ToolCallSaver
from qs.tokenizers.qwen import QwenTokenizer
from qs.utils.file import read_json_from_path


class DummyCalledTool:
    def __init__(self, name: str, call_id: str, arguments: dict):
        self.name = name
        self.call_id = call_id
        self.arguments = arguments
        self.tool_call = ToolCall(
            id=call_id, function=FunctionCall(name=name, arguments=arguments)
        )


class DummyCalledDeepSkill:
    def __init__(self, name: str, call_id: str, arguments: dict):
        self.name = name
        self.call_id = call_id
        self.arguments = arguments
        self.tool_call = ToolCall(
            id=call_id, function=FunctionCall(name=name, arguments=arguments)
        )


class DummyCalledExitTool:
    def __init__(self, name: str, call_id: str, arguments: dict):
        self.name = name
        self.call_id = call_id
        self.arguments = arguments
        self.tool_call = ToolCall(
            id=call_id, function=FunctionCall(name=name, arguments=arguments)
        )


# --- ToolCallData Tests ---


def test_tool_call_data_valid():
    data = ToolCallData(
        name="test_tool",
        id="call-001",
        args={"arg1": "val1"},
        result={"output": 42},
    )
    assert data.name == "test_tool"
    assert data.id == "call-001"
    assert data.args == {"arg1": "val1"}
    assert data.result == {"output": 42}


@pytest.mark.parametrize(
    ("name", "call_id", "args", "error"),
    [
        (123, "id-1", {}, TypeError),
        ("", "id-1", {}, ValueError),
        ("   ", "id-1", {}, ValueError),
        ("tool", 123, {}, TypeError),
        ("tool", "", {}, ValueError),
        ("tool", "   ", {}, ValueError),
        ("tool", "id-1", "not-a-dict", TypeError),
    ],
)
def test_tool_call_data_invalid_fields(name, call_id, args, error):
    with pytest.raises(error):
        ToolCallData(name=name, id=call_id, args=args, result="result")


def test_tool_call_data_to_and_from_dict():
    original = ToolCallData(
        name="weather",
        id="w_123",
        args={"city": "Rome"},
        result="Sunny, 25C",
    )
    raw = original.to_dict()
    assert raw == {
        "name": "weather",
        "id": "w_123",
        "args": {"city": "Rome"},
        "result": "Sunny, 25C",
        "skill_path": "/",
        "tool_call_id": None,
        "thinking": None,
    }
    rebuilt = ToolCallData.from_dict(raw)
    assert rebuilt == original


def test_tool_call_data_from_dict_invalid():
    with pytest.raises(TypeError, match="must be a dictionary"):
        ToolCallData.from_dict("invalid")  # type: ignore

    with pytest.raises(ValueError, match="Missing required keys"):
        ToolCallData.from_dict({"name": "foo", "id": "bar"})


# --- ToolCallSaver Tests ---


def test_saver_path_resolution_with_identity(tmp_path: Path):
    identity = "custom/test-model:v1"
    expected_hash = sha256(identity.encode("utf-8")).hexdigest()
    saver = ToolCallSaver(
        tokenizer_identity=identity,
        cache_directory=tmp_path,
        folder_name="my_calls",
    )
    expected_dir = tmp_path / expected_hash / "temp" / "my_calls"
    assert saver.directory == expected_dir
    assert expected_dir.is_dir()


def test_saver_path_resolution_with_tokenizer_instance(tmp_path: Path):
    tokenizer = QwenTokenizer(cache_directory=tmp_path)
    expected_hash = sha256(tokenizer._identity.encode("utf-8")).hexdigest()
    saver = ToolCallSaver(
        tokenizer=tokenizer,
        cache_directory=tmp_path,
    )
    assert saver.directory == tmp_path / expected_hash / "temp" / "tool_calls"


def test_saver_invalid_inputs():
    with pytest.raises(ValueError, match="Either 'directory'"):
        ToolCallSaver()

    with pytest.raises(TypeError, match="directory must be a Path"):
        ToolCallSaver(directory="not-a-path")  # type: ignore

    with pytest.raises(ValueError, match="folder_name must be a non-empty string"):
        ToolCallSaver(tokenizer_identity="ident", folder_name="")


def test_saver_record_and_save(tmp_path: Path):
    saver = ToolCallSaver(directory=tmp_path)
    record = saver.record(
        name="calculator",
        id="call_999",
        args={"a": 10, "b": 20},
        result=30,
    )

    assert record.name == "calculator"
    assert record.id == "call_999"

    saved_file = tmp_path / "call_999.json"
    assert saved_file.is_file()
    data = read_json_from_path(saved_file)
    assert data["result"] == 30


def test_saver_record_call_generics(tmp_path: Path):
    saver = ToolCallSaver(directory=tmp_path)

    # 1. Normal tool call
    called_tool = DummyCalledTool(name="bash", call_id="t_1", arguments={"cmd": "ls"})
    res1 = saver.record_call(called_tool, result="file1.txt", id="t_1")
    assert res1.name == "bash"
    assert res1.id == "t_1"
    assert res1.args == {"cmd": "ls"}

    # 2. Deep skill call
    deep_skill = DummyCalledDeepSkill(
        name="deep_code", call_id="ds_1", arguments={"prompt": "refactor"}
    )
    res2 = saver.record_call(deep_skill, result="instructions rendered", id="ds_1")
    assert res2.name == "deep_code"
    assert res2.id == "ds_1"
    assert res2.args == {"prompt": "refactor"}

    # 3. Exit tool call
    exit_tool = DummyCalledExitTool(
        name="finish", call_id="exit_1", arguments={"message": "complete"}
    )
    res3 = saver.record_call(exit_tool, result={"status": "done"}, id="exit_1")
    assert res3.name == "finish"
    assert res3.id == "exit_1"

    # 4. Raw ToolCall with custom ID override
    raw_call = ToolCall(
        id="orig_id", function=FunctionCall(name="echo", arguments={"text": "hi"})
    )
    res4 = saver.record_call(raw_call, result="hi", id="override_id")
    assert res4.name == "echo"
    assert res4.id == "override_id"

    # 5. Invalid tool call missing name
    with pytest.raises(TypeError, match="implement CalledToolMetadata"):
        saver.record_call(object(), result="none", id="invalid")


# --- ToolCallReader Tests ---


def test_reader_read_and_get(tmp_path: Path):
    saver = ToolCallSaver(directory=tmp_path)
    reader = ToolCallReader(directory=tmp_path)

    saver.record(
        name="fetch", id="call_abc", args={"url": "https://example.com"}, result=200
    )

    assert reader.exists("call_abc")
    assert not reader.exists("non_existent")

    record = reader.read("call_abc")
    assert record is not None
    assert record.name == "fetch"
    assert record.args == {"url": "https://example.com"}
    assert record.result == 200

    assert reader.read("non_existent") is None
    with pytest.raises(FileNotFoundError, match="No tool call recorded"):
        reader.get("non_existent")


def test_reader_list_ids_and_read_all(tmp_path: Path):
    saver = ToolCallSaver(directory=tmp_path)
    reader = ToolCallReader(directory=tmp_path)

    for i in range(5):
        saver.record(name=f"tool_{i}", id=f"id_{i}", args={"i": i}, result=i * 2)

    ids = reader.list_ids()
    assert sorted(ids) == [f"id_{i}" for i in range(5)]

    all_records = reader.read_all()
    assert len(all_records) == 5
    assert {r.id for r in all_records} == {f"id_{i}" for i in range(5)}


def test_reader_handles_corrupted_file(tmp_path: Path):
    saver = ToolCallSaver(directory=tmp_path)
    reader = ToolCallReader(directory=tmp_path)

    saver.record(name="valid", id="valid_id", args={}, result="ok")

    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("invalid json content!", encoding="utf-8")

    # read_all skips corrupted files gracefully
    all_records = reader.read_all()
    assert len(all_records) == 1
    assert all_records[0].id == "valid_id"

    # list_ids exposes only valid, readable records
    ids = reader.list_ids()
    assert "corrupt" not in ids
    assert "valid_id" in ids


def test_concurrent_saves_and_reads(tmp_path: Path):
    saver = ToolCallSaver(directory=tmp_path)
    reader = ToolCallReader(directory=tmp_path)

    def worker(i: int):
        return saver.record(
            name=f"t_{i}", id=f"call_{i:04d}", args={"step": i}, result={"res": i}
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(worker, range(100)))

    assert len(records) == 100
    assert len(reader.read_all()) == 100
    assert len(reader.list_ids()) == 100
    for r in records:
        assert reader.get(r.id) == r
