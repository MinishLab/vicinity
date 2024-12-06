from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

from numpy import typing as npt
from voyager import Index, Space

from vicinity.backends.base import AbstractBackend, BaseArgs
from vicinity.datatypes import Backend, QueryResult
from vicinity.utils import Metric


@dataclass
class VoyagerArgs(BaseArgs):
    dim: int = 0
    metric: Metric = Metric.COSINE
    ef_construction: int = 200
    m: int = 16


class VoyagerBackend(AbstractBackend[VoyagerArgs]):
    argument_class = VoyagerArgs
    supported_metrics = {Metric.COSINE, Metric.EUCLIDEAN}
    _metric_to_space = {
        Metric.COSINE: Space.Cosine,
        Metric.EUCLIDEAN: Space.Euclidean,
    }

    def __init__(
        self,
        index: Index,
        arguments: VoyagerArgs,
    ) -> None:
        """Initialize the backend using vectors."""
        super().__init__(arguments)
        self.index = index

    @classmethod
    def from_vectors(
        cls: type[VoyagerBackend],
        vectors: npt.NDArray,
        metric: Union[str, Metric],
        ef_construction: int,
        m: int,
        **kwargs: Any,
    ) -> VoyagerBackend:
        """Create a new instance from vectors."""
        metric_enum = Metric.from_string(metric)

        if metric_enum not in cls.supported_metrics:
            raise ValueError(f"Metric '{metric_enum.value}' is not supported by VoyagerBackend.")

        space = cls._metric_to_space[metric_enum]
        dim = vectors.shape[1]
        index = Index(
            space=space,
            num_dimensions=dim,
            M=m,
            ef_construction=ef_construction,
        )
        index.add_items(vectors)
        return cls(
            index,
            VoyagerArgs(dim=dim, metric=metric_enum, ef_construction=ef_construction, m=m),
        )

    def query(self, query: npt.NDArray, k: int) -> QueryResult:
        """Query the backend for the nearest neighbors."""
        indices, distances = self.index.query(query, k)
        return list(zip(indices, distances))

    @classmethod
    def load(cls: type[VoyagerBackend], base_path: Path) -> VoyagerBackend:
        """Load the vectors from a path."""
        path = Path(base_path) / "index.bin"
        arguments = VoyagerArgs.load(base_path / "arguments.json")
        index = Index.load(str(path))
        return cls(index, arguments=arguments)

    def save(self, base_path: Path) -> None:
        """Save the vectors to a path."""
        path = Path(base_path) / "index.bin"
        self.index.save(str(path))
        self.arguments.dump(base_path / "arguments.json")

    def insert(self, vectors: npt.NDArray) -> None:
        """Insert vectors into the backend."""
        self.index.add_items(vectors)

    def delete(self, indices: list[int]) -> None:
        """Delete vectors from the backend."""
        raise NotImplementedError("Deletion is not supported in Voyager backend.")

    def threshold(self, vectors: npt.NDArray, threshold: float) -> list[npt.NDArray]:
        """Threshold the backend."""
        out: list[npt.NDArray] = []
        for x, y in self.query(vectors, len(self)):
            out.append(x[y < threshold])

        return out

    @property
    def backend_type(self) -> Backend:
        """The type of the backend."""
        return Backend.VOYAGER

    @property
    def dim(self) -> int:
        """Get the dimension of the space."""
        return self.index.num_dimensions

    def __len__(self) -> int:
        """Get the number of vectors."""
        return self.index.num_elements
