"""Shared error results for unavailable tool dependencies."""


def missing_dependency(error: ModuleNotFoundError) -> dict[str, object]:
    return {
        "ok": False,
        "error_kind": "missing_dependency",
        "error": f"Missing exmekh_lib runtime module: {error.name}.",
        "hint": "Install the ExMekh-Lib wheel with its declared dependencies in this environment.",
    }
