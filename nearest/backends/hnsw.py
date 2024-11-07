from __future__ import annotations

from pathlib import Path
from typing import Literal

from hnswlib import Index as HnswIndex
from numpy import typing as npt

from nearest.backends.base import BaseBackend
from nearest.datatypes import QueryResult


class HnswBackend(BaseBackend):
    def __init__(
        self,
        vectors: npt.NDArray,
        space: Literal["cosine", "l2", "ip"] = "cosine",
        ef_construction: int = 200,
        m: int = 16,
    ) -> None:
        """Initialize the backend using vectors."""
        self.index = HnswIndex(space=space, dim=vectors.shape[1])
        self.index.init_index(max_elements=vectors.shape[0], ef_construction=ef_construction, M=m)
        self.index.add_items(vectors)
        self._dim = vectors.shape[1]

    @property
    def dim(self) -> int:
        """Get the dimension of the space."""
        return self._dim

    def __len__(self) -> int:
        """Get the number of vectors."""
        return self.index.get_current_count()

    @classmethod
    def load(cls: type[HnswBackend], path: Path) -> HnswBackend:
        """Load the vectors from a path."""
        index = HnswIndex.load_index(path)
        return cls(index)

    def save(self, base_path: Path) -> None:
        """Save the vectors to a path."""
        path = Path(base_path) / "index.bin"
        self.index.save_index(path)

    def query(self, vectors: npt.NDArray, k: int) -> QueryResult:
        """Query the backend."""
        return list(zip(*self.index.knn_query(vectors, k)))

    def insert(self, vectors: npt.NDArray) -> None:
        """Insert vectors into the backend."""
        self.index.add_items(vectors)

    def delete(self, indices: list[int]) -> None:
        """Delete vectors from the backend."""
        self.index.remove_items(indices)

    def threshold(self, vectors: npt.NDArray, threshold: float) -> list[list[int]]:
        """Threshold the backend."""
        raise NotImplementedError()
