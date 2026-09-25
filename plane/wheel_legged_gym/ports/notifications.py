"""Notification callback boundary; workflows need only a callable, not HAPI."""
from typing import Protocol


class NotificationSink(Protocol):
    def __call__(self, message: str) -> None:
        ...
