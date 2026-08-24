import base64
from typing import cast

import pytest

from qs.utils.file import convert_bytes_to_base64


@pytest.mark.parametrize(
    ("file_bytes", "expected"),
    [
        (b"", ""),
        (b"hello", "aGVsbG8="),
        (bytes(range(8)), "AAECAwQFBgc="),
    ],
)
def test_convert_bytes_to_base64_returns_utf8_base64(file_bytes, expected) -> None:
    result = convert_bytes_to_base64(file_bytes)

    assert result == expected
    assert base64.b64decode(result) == file_bytes


def test_convert_bytes_to_base64_rejects_non_bytes() -> None:
    with pytest.raises(TypeError, match="Input must be of type 'bytes'"):
        convert_bytes_to_base64(cast(bytes, "content"))
