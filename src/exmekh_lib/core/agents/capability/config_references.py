"""Resolve explicit configuration references before parsing capability Markdown."""

import re

from exmekh_lib.config import file_editing, file_inspection, reasoning, runtime

_REFERENCE = re.compile(r"{{config\.([A-Z_]+)\.([a-z][a-z0-9_]*)}}")


def render_config_references(text: str) -> str:
    settings = {
        "FILE_EDITING": file_editing.FILE_EDITING,
        "FILE_INSPECTION": file_inspection.FILE_INSPECTION,
        "DATE_TIME": reasoning.DATE_TIME,
        "RUNTIME": runtime.RUNTIME,
    }

    def replace(match: re.Match[str]) -> str:
        name, field = match.groups()
        configuration = settings.get(name)
        value = getattr(configuration, field, None)
        if not isinstance(value, (str, int, float)):
            raise ValueError(
                f"Unknown or non-scalar configuration reference: {name}.{field}"
            )
        return str(value)

    return _REFERENCE.sub(replace, text)
