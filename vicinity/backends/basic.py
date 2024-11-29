from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np
from numpy import typing as npt

from vicinity.backends.base import AbstractBackend, BaseArgs
from vicinity.datatypes import Backend, Matrix, QueryResult
from vicinity.utils import Metric, normalize, normalize_or_copy


@dataclass
class BasicArgs(BaseArgs):
    metric: str = "cosine"


class BasicBackend(AbstractBackend[BasicArgs], ABC):
    argument_class = BasicArgs
    _vectors: npt.NDArray
    supported_metrics = {Metric.COSINE, Metric.EUCLIDEAN}

    def __init__(self, arguments: BasicArgs) -> None:
        """Initialize the backend."""
        super().__init__(arguments)

    def __len__(self) -> int:
        """Get the number of vectors."""
        return self.vectors.shape[0]

    @property
    def backend_type(self) -> Backend:
        """The type of the backend."""
        return Backend.BASIC

    @property
    def dim(self) -> int:
        """The size of the space."""
        return self.vectors.shape[1]

    @property
    def vectors(self) -> npt.NDArray:
        """The vectors themselves."""
        return self._vectors

    @vectors.setter
    def vectors(self, x: Matrix) -> None:
        """Set the vectors."""
        matrix = np.asarray(x)
        if np.ndim(matrix) != 2:
            raise ValueError(f"Your array does not have 2 dimensions: {np.ndim(matrix)}")
        self._vectors = matrix
        self._update_precomputed_data()

    @abstractmethod
    def _update_precomputed_data(self) -> None:
        """Update precomputed data based on the metric."""
        raise NotImplementedError()

    @abstractmethod
    def _dist(self, x: npt.NDArray) -> npt.NDArray:
        """Compute distances between x and self._vectors based on the metric."""
        raise NotImplementedError()

    @classmethod
    def from_vectors(
        cls: type[BasicBackend],
        vectors: npt.NDArray,
        metric: Union[str, Metric],
        **kwargs: Any,
    ) -> BasicBackend:
        """Create a new instance from vectors."""
        metric_enum = Metric.from_string(metric)

        if metric_enum not in cls.supported_metrics:
            raise ValueError(f"Metric '{metric_enum.value}' is not supported by BasicBackend.")

        if metric_enum == Metric.COSINE:
            return CosineBasicBackend(vectors, BasicArgs(metric=metric_enum.value))
        elif metric_enum == Metric.EUCLIDEAN:
            return EuclideanBasicBackend(vectors, BasicArgs(metric=metric_enum.value))
        else:
            raise ValueError(f"Unsupported metric: {metric_enum.value}")

    @classmethod
    def load(cls, folder: Path) -> BasicBackend:
        """Load the vectors from a path."""
        path = folder / "vectors.npy"
        arguments = BasicArgs.load(folder / "arguments.json")
        metric_enum = Metric.from_string(arguments.metric)

        if metric_enum == Metric.COSINE:
            return CosineBasicBackend(np.load(path), arguments)
        elif metric_enum == Metric.EUCLIDEAN:
            return EuclideanBasicBackend(np.load(path), arguments)
        else:
            raise ValueError(f"Unsupported metric: {metric_enum.value}")

    def save(self, folder: Path) -> None:
        """Save the vectors to a path."""
        path = folder / "vectors.npy"
        self.arguments.dump(folder / "arguments.json")
        with open(path, "wb") as f:
            np.save(f, self._vectors)

    def threshold(
        self,
        vectors: npt.NDArray,
        threshold: float,
    ) -> list[npt.NDArray]:
        """Batched distance thresholding."""
        out: list[npt.NDArray] = []
        for i in range(0, len(vectors), 1024):
            batch = vectors[i : i + 1024]
            distances = self._dist(batch)
            for sims in distances:
                indices = np.flatnonzero(sims <= threshold)
                sorted_indices = indices[np.argsort(sims[indices])]
                out.append(sorted_indices)
        return out

    def query(
        self,
        vectors: npt.NDArray,
        k: int,
    ) -> QueryResult:
        """Batched distance query."""
        if k < 1:
            raise ValueError(f"k should be >= 1, is now {k}")

        out: QueryResult = []
        num_vectors = len(self.vectors)
        effective_k = min(k, num_vectors)

        for index in range(0, len(vectors), 1024):
            batch = vectors[index : index + 1024]
            distances = self._dist(batch)

            indices = np.argpartition(distances, kth=effective_k - 1, axis=1)[:, :effective_k]
            sorted_indices = np.take_along_axis(
                indices, np.argsort(np.take_along_axis(distances, indices, axis=1)), axis=1
            )
            sorted_distances = np.take_along_axis(distances, sorted_indices, axis=1)

            out.extend(zip(sorted_indices, sorted_distances))

        return out

    def insert(self, vectors: npt.NDArray) -> None:
        """Insert vectors into the vector space."""
        self._vectors = np.vstack([self._vectors, vectors])
        self._update_precomputed_data()

    def delete(self, indices: list[int]) -> None:
        """Deletes specific indices from the vector space."""
        self._vectors = np.delete(self._vectors, indices, axis=0)
        self._update_precomputed_data()


class CosineBasicBackend(BasicBackend):
    def __init__(self, vectors: npt.NDArray, arguments: BasicArgs) -> None:
        """Initialize the cosine basic backend."""
        super().__init__(arguments)
        self._vectors = vectors
        self._norm_vectors: npt.NDArray | None = None
        self._update_precomputed_data()

    def _update_precomputed_data(self) -> None:
        """Update precomputed data for cosine similarity."""
        self._norm_vectors = normalize_or_copy(self._vectors)

    @property
    def norm_vectors(self) -> npt.NDArray:
        """Return normalized vectors."""
        if self._norm_vectors is None:
            self._norm_vectors = normalize_or_copy(self._vectors)
        return self._norm_vectors

    def _dist(self, x: npt.NDArray) -> npt.NDArray:
        """Compute cosine distance."""
        x_norm = normalize(x)
        sim = x_norm.dot(self.norm_vectors.T)
        return 1 - sim


class EuclideanBasicBackend(BasicBackend):
    def __init__(self, vectors: npt.NDArray, arguments: BasicArgs) -> None:
        """Initialize the Euclidean basic backend."""
        super().__init__(arguments)
        self._vectors = vectors
        self._squared_norm_vectors: npt.NDArray | None = None
        self._update_precomputed_data()

    def _update_precomputed_data(self) -> None:
        """Update precomputed data for Euclidean distance."""
        self._squared_norm_vectors = (self._vectors**2).sum(1)

    @property
    def squared_norm_vectors(self) -> npt.NDArray:
        """Return squared norms of vectors."""
        if self._squared_norm_vectors is None:
            self._squared_norm_vectors = (self._vectors**2).sum(1)
        return self._squared_norm_vectors

    def _dist(self, x: npt.NDArray) -> npt.NDArray:
        """Compute Euclidean distance."""
        x_norm = (x**2).sum(1)
        dists_squared = (x_norm[:, None] + self.squared_norm_vectors[None, :]) - 2 * (x @ self._vectors.T)
        dists_squared = np.clip(dists_squared, 0, None)  # Ensure non-negative distances
        return np.sqrt(dists_squared)
