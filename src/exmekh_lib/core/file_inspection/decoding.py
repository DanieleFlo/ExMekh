"""Decode physical text consistently for inspection and line reads."""

import codecs
import io
import tokenize


def decode_text(raw: bytes, extension: str) -> tuple[str, str] | None:
    if extension in {".py", ".pyi"}:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
        return raw.decode(encoding), encoding
    for marker, encoding in (
        (codecs.BOM_UTF32_LE, "utf-32"),
        (codecs.BOM_UTF32_BE, "utf-32"),
        (codecs.BOM_UTF16_LE, "utf-16"),
        (codecs.BOM_UTF16_BE, "utf-16"),
        (codecs.BOM_UTF8, "utf-8-sig"),
    ):
        if raw.startswith(marker):
            return raw.decode(encoding), encoding
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if any(ord(character) < 32 and character not in "\t\n\r\f" for character in text):
        return None
    return text, "utf-8"
