from dataclasses import FrozenInstanceError

import pytest

from exmekh_lib.core.models.tool_output_reference import ToolOutputReference


def test_references_preserve_opaque_identifiers_and_are_immutable():
    data = {"name": "work", "id": "id\x00<&>"}
    reference = ToolOutputReference.from_dict(data)
    data["id"] = "changed"
    exported = reference.to_dict()
    exported["id"] = "changed"

    assert reference.to_dict() == {"name": "work", "id": "id\x00<&>"}
    with pytest.raises(FrozenInstanceError):
        reference.id = "changed"


@pytest.mark.parametrize(
    ("data", "error"),
    [
        ([], TypeError),
        ({"name": "work"}, ValueError),
        ({"name": "work", "id": "output", "args": {}}, ValueError),
        ({"name": 1, "id": "output"}, TypeError),
        ({"name": "work", "id": None}, TypeError),
        ({"name": " ", "id": "output"}, ValueError),
        ({"name": "work", "id": ""}, ValueError),
    ],
)
def test_references_reject_incomplete_or_invalid_metadata(data, error):
    with pytest.raises(error):
        ToolOutputReference.from_dict(data)
