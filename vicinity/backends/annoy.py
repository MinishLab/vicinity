from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np
from annoy import AnnoyIndex
from numpy import typing as npt

from vicinity.backends.base import AbstractBackend, BaseArgs
from vicinity.datatypes import Backend, QueryResult
from vicinity.utils import Metric, normalize


@dataclass
class AnnoyArgs(BaseArgs):
    dim: int = 0
    metric: str = "cosine"
    trees: int = 100
    length: int | None = None


class AnnoyBackend(AbstractBackend[AnnoyArgs]):
    argument_class = AnnoyArgs
    supported_metrics = {Metric.COSINE, Metric.EUCLIDEAN, Metric.INNER_PRODUCT}
    inverse_metric_mapping = {
        Metric.COSINE: "dot",
        Metric.EUCLIDEAN: "euclidean",
        Metric.INNER_PRODUCT: "dot",
    }

    def __init__(
        self,
        index: AnnoyIndex,
        arguments: AnnoyArgs,
    ) -> None:
        """Initialize the backend using vectors."""
        super().__init__(arguments)
        self.index = index
        if arguments.length is None:
            raise ValueError("Length must be provided.")
        self.length = arguments.length

    @classmethod
    def from_vectors(
        cls: type[AnnoyBackend],
        vectors: npt.NDArray,
        metric: Union[str, Metric],
        trees: int,
        **kwargs: Any,
    ) -> AnnoyBackend:
        """Create a new instance from vectors."""
        metric_enum = Metric.from_string(metric)

        if metric_enum not in cls.supported_metrics:
            raise ValueError(f"Metric '{metric_enum.value}' is not supported by AnnoyBackend.")

        metric = cls._map_metric_to_string(metric_enum)

        if metric == "dot":
            vectors = normalize(vectors)

        dim = vectors.shape[1]
        index = AnnoyIndex(f=dim, metric=metric)  # type: ignore
        for i, vector in enumerate(vectors):
            index.add_item(i, vector)
        index.build(trees)

        arguments = AnnoyArgs(dim=dim, metric=metric, trees=trees, length=len(vectors))  # type: ignore
        return AnnoyBackend(index, arguments=arguments)

    @property
    def backend_type(self) -> Backend:
        """The type of the backend."""
        return Backend.ANNOY

    @property
    def dim(self) -> int:
        """Get the dimension of the space."""
        return self.index.f

    def __len__(self) -> int:
        """Get the number of vectors."""
        return self.length

    @classmethod
    def load(cls: type[AnnoyBackend], base_path: Path) -> AnnoyBackend:
        """Load the vectors from a path."""
        path = Path(base_path) / "index.bin"
        arguments = AnnoyArgs.load(base_path / "arguments.json")
        index = AnnoyIndex(arguments.dim, arguments.metric)  # type: ignore
        index.load(str(path))

        return cls(index, arguments=arguments)

    def save(self, base_path: Path) -> None:
        """Save the vectors to a path."""
        path = Path(base_path) / "index.bin"
        self.index.save(str(path))
        # Ensure the length is set before saving
        self.arguments.length = len(self)
        self.arguments.dump(base_path / "arguments.json")

    def query(self, vectors: npt.NDArray, k: int) -> QueryResult:
        """Query the backend."""
        out = []
        for vec in vectors:
            if self.arguments.metric == "dot":
                vec = normalize(vec)
            indices, scores = self.index.get_nns_by_vector(vec, k, include_distances=True)
            scores_array = np.asarray(scores)
            if self.arguments.metric == "dot":
                # Convert cosine similarity to cosine distance
                scores_array = 1 - scores_array
            out.append((np.asarray(indices), scores_array))
        return out

    def insert(self, vectors: npt.NDArray) -> None:
        """Insert vectors into the backend."""
        raise NotImplementedError("Insertion is not supported in Annoy backend.")

    def delete(self, indices: list[int]) -> None:
        """Delete vectors from the backend."""
        raise NotImplementedError("Deletion is not supported in Annoy backend.")

    def threshold(self, vectors: npt.NDArray, threshold: float) -> list[npt.NDArray]:
        """Threshold the backend."""
        out: list[npt.NDArray] = []
        for x, y in self.query(vectors, 100):
            out.append(x[y < threshold])
        return out
