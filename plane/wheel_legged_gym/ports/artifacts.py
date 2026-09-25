"""Minimal artifact access for evaluation workflows; no filesystem implementation."""
from pathlib import Path
from typing import Any, Protocol


class EvaluationArtifacts(Protocol):
    directory: Path

    def exists(self, path: Path) -> bool: ...

    def read_json(self, path: Path) -> Any: ...

    def write_json(self, name: str, value: Any) -> None: ...
