from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np
from numpy import typing as npt
from voyager import Index, Space

from vicinity.backends.base import AbstractBackend, BaseArgs
from vicinity.datatypes import Backend, QueryResult
from vicinity.utils import Metric, normalize


@dataclass
class VoyagerArgs(BaseArgs):
    dim: int = 0
    metric: str = "cosine"
    ef_construction: int = 200
    m: int = 16


class VoyagerBackend(AbstractBackend[VoyagerArgs]):
    argument_class = VoyagerArgs
    supported_metrics = {Metric.COSINE, Metric.EUCLIDEAN}
    inverse_metric_mapping = {
        Metric.COSINE: "cosine",
        Metric.EUCLIDEAN: "l2",
    }

    metric_mapping = {
        "l2": 0,
        "cosine": 2,
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
            raise ValueError(
                f"Metric '{metric_enum.value}' is not supported by VoyagerBackend."
            )

        metric = cls._map_metric_to_string(metric_enum)
        dim = vectors.shape[1]
        space = Space(value=cls.metric_mapping[metric])
        index = Index(
            space=space,
            num_dimensions=dim,
            M=m,
            ef_construction=ef_construction,
        )
        index.add_items(vectors)
        return cls(
            index,
            VoyagerArgs(dim=dim, metric=metric, ef_construction=ef_construction, m=m),
        )

    def query(self, query: npt.NDArray, k: int) -> QueryResult:
        """Query the backend for the nearest neighbors."""
        query = normalize(query)
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
        """Delete vectors from the backend.
        Note that ids are marked as deleted, won't show up in results,
        but they would still take up space.
        Refer: https://spotify.github.io/voyager/python/reference.html#voyager.Index.mark_deleted
        """
        for _id in indices:
            self.index.mark_deleted(_id)

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
