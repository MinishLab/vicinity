from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from hnswlib import Index as HnswIndex
from numpy import typing as npt

from nearest.backends.base import AbstractBackend, BaseArgs
from nearest.datatypes import Backend, QueryResult


@dataclass(frozen=True)
class HNSWArgs(BaseArgs):
    dim: int
    space: Literal["cosine", "l2", "ip"] = "cosine"
    ef_construction: int = 200
    m: int = 16


class HNSWBackend(AbstractBackend):
    argument_class = HNSWArgs

    def __init__(
        self,
        index: HnswIndex,
        arguments: HNSWArgs,
    ) -> None:
        """Initialize the backend using vectors."""
        super().__init__(arguments)
        self.index = index

    @classmethod
    def from_vectors(
        cls: type[HNSWBackend],
        vectors: npt.NDArray,
        dim: int,
        space: Literal["cosine", "l2", "ip"],
        ef_construction: int,
        m: int,
    ) -> HNSWBackend:
        """Create a new instance from vectors."""
        index = HnswIndex(space=space, dim=vectors.shape[1])
        index.init_index(max_elements=vectors.shape[0], ef_construction=ef_construction, M=m)
        index.add_items(vectors)
        arguments = HNSWArgs(dim=dim, space=space, ef_construction=ef_construction, m=m)
        return HNSWBackend(index, arguments=arguments)

    @property
    def backend_type(self) -> Backend:
        """The type of the backend."""
        return Backend.HNSW

    @property
    def dim(self) -> int:
        """Get the dimension of the space."""
        return self.index.dim

    def __len__(self) -> int:
        """Get the number of vectors."""
        return self.index.get_current_count()

    @classmethod
    def load(cls: type[HNSWBackend], base_path: Path) -> HNSWBackend:
        """Load the vectors from a path."""
        path = Path(base_path) / "index.bin"
        arguments = HNSWArgs.load(base_path / "arguments.json")
        index = HnswIndex(space=arguments.space, dim=arguments.dim)
        index.load_index(str(path))
        return cls(index, arguments=arguments)

    def save(self, base_path: Path) -> None:
        """Save the vectors to a path."""
        path = Path(base_path) / "index.bin"
        self.index.save_index(str(path))
        self.arguments.dump(base_path / "arguments.json")

    def query(self, vectors: npt.NDArray, k: int) -> QueryResult:
        """Query the backend."""
        return list(zip(*self.index.knn_query(vectors, k)))

    def insert(self, vectors: npt.NDArray) -> None:
        """Insert vectors into the backend."""
        self.index.add_items(vectors)

    def delete(self, indices: list[int]) -> None:
        """Delete vectors from the backend."""
        for index in indices:
            self.index.mark_deleted(index)

    def threshold(self, vectors: npt.NDArray, threshold: float) -> list[npt.NDArray]:
        """Threshold the backend."""
        out: list[npt.NDArray] = []
        for x, y in self.query(vectors, 100):
            out.append(x[y < threshold])

        return out
