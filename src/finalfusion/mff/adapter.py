"""MFF-specific RAW adapter."""

from ..common import TaskInputAdapter


class MffInputAdapter(TaskInputAdapter):
    def __init__(self) -> None:
        super().__init__(shared_selector=False)
