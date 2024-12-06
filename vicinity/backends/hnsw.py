from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

from hnswlib import Index as HnswIndex
from numpy import typing as npt

from vicinity.backends.base import AbstractBackend, BaseArgs
from vicinity.datatypes import Backend, QueryResult
from vicinity.utils import Metric


@dataclass
class HNSWArgs(BaseArgs):
    dim: int = 0
    metric: Metric = Metric.COSINE
    ef_construction: int = 200
    m: int = 16


class HNSWBackend(AbstractBackend[HNSWArgs]):
    argument_class = HNSWArgs
    supported_metrics = {Metric.COSINE, Metric.EUCLIDEAN}
    inverse_metric_mapping = {
        Metric.COSINE: "cosine",
        Metric.EUCLIDEAN: "l2",
    }

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
        metric: Union[str, Metric],
        ef_construction: int,
        m: int,
        **kwargs: Any,
    ) -> HNSWBackend:
        """Create a new instance from vectors."""
        metric_enum = Metric.from_string(metric)

        if metric_enum not in cls.supported_metrics:
            raise ValueError(f"Metric '{metric_enum.value}' is not supported by HNSWBackend.")

        # Map Metric to HNSW's space parameter
        metric = cls._map_metric_to_string(metric_enum)
        dim = vectors.shape[1]
        index = HnswIndex(space=metric, dim=dim)
        index.init_index(max_elements=vectors.shape[0], ef_construction=ef_construction, M=m)
        index.add_items(vectors)
        arguments = HNSWArgs(dim=dim, metric=metric_enum, ef_construction=ef_construction, m=m)
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
        mapped_metric = cls.inverse_metric_mapping[arguments.metric]
        index = HnswIndex(space=mapped_metric, dim=arguments.dim)
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
        raise NotImplementedError("Deletion is not supported in HNSW backend.")

    def threshold(self, vectors: npt.NDArray, threshold: float) -> list[npt.NDArray]:
        """Threshold the backend."""
        out: list[npt.NDArray] = []
        for x, y in self.query(vectors, 100):
            out.append(x[y < threshold])

        return out
