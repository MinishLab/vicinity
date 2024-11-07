from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar

from numpy import typing as npt

from nearest.datatypes import QueryResult


class BaseBackend(ABC):
    @abstractmethod
    def __init__(self, vectors: npt.NDArray, **kwargs: Any) -> None:
        """Initialize the backend with vectors."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """The number of items in the backend."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def dim(self) -> int:
        """The size of the space."""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def load(cls: type[BaseType], path: Path) -> BaseType:
        """Load a backend from a file."""
        raise NotImplementedError()

    @abstractmethod
    def save(self, base_path: Path) -> None:
        """Save the backend to a file."""
        raise NotImplementedError()

    @abstractmethod
    def insert(self, vectors: npt.NDArray) -> None:
        """Insert vectors into the backend."""
        raise NotImplementedError()

    @abstractmethod
    def delete(self, indices: list[int]) -> None:
        """Delete vectors from the backend."""
        raise NotImplementedError()

    @abstractmethod
    def threshold(self, vectors: npt.NDArray, threshold: float) -> list[list[int]]:
        """Threshold the backend."""
        raise NotImplementedError()

    @abstractmethod
    def query(self, vectors: npt.NDArray, k: int) -> QueryResult:
        """Query the backend."""
        raise NotImplementedError()


BaseType = TypeVar("BaseType", bound=BaseBackend)
