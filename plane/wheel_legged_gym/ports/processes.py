"""Process capabilities required by workflows; implemented at application boundaries."""
from typing import Optional, Mapping, Protocol, Sequence, Union
from pathlib import Path


class PythonJob(Protocol):
    def __call__(self, arguments: Sequence[Union[str, Path]], tag: str,
                 extra: Optional[Mapping[str, str]] = None) -> None:
        """Complete a logged Python job or raise; never silently ignore failures."""
        ...
