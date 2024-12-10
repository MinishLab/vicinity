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
    metric: Metric = Metric.COSINE


class BasicVectorStore:
    def __init__(self, *, vectors: npt.NDArray, **kwargs: Any) -> None:
        """
        A basic vector store that just stores vectors.

        Note that we use kwargs in order to use this class as a mixin.

        :param vectors: The vectors to store.
        :param **kwargs: Additional arguments. These are passed on to the super class.
        """
        super().__init__(**kwargs)
        self._vectors = vectors

    def _update_precomputed_data(self) -> None:
        """Update precomputed data based on the metric."""
        # NOTE: this is a no-op in the base implementation.
        return

    def get_by_index(self, indices: list[int]) -> npt.NDArray:
        """Get vectors by index."""
        return self._vectors[indices]

    def insert(self, vectors: npt.NDArray) -> None:
        """Insert vectors into the vector space."""
        self._vectors = np.vstack([self._vectors, vectors])
        self._update_precomputed_data()

    def delete(self, indices: list[int]) -> None:
        """Deletes specific indices from the vector space."""
        self._vectors = np.delete(self._vectors, indices, axis=0)
        self._update_precomputed_data()

    def save(self, folder: Path) -> None:
        """Save the vectors to a path."""
        path = folder / "vectors.npy"
        with open(path, "wb") as f:
            np.save(f, self._vectors)

    @staticmethod
    def _load_vectors(folder: Path) -> npt.NDArray:
        """Load the vectors from a path."""
        path = folder / "vectors.npy"
        with open(path, "rb") as f:
            vectors = np.load(f)

        return vectors

    @classmethod
    def load(cls, folder: Path) -> BasicVectorStore:
        """Load the vectors from a path."""
        vectors = cls._load_vectors(folder)
        return cls(vectors=vectors)

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


class BasicBackend(BasicVectorStore, AbstractBackend[BasicArgs], ABC):
    argument_class = BasicArgs
    _vectors: npt.NDArray
    supported_metrics = {Metric.COSINE, Metric.EUCLIDEAN}

    def __init__(self, vectors: npt.NDArray, arguments: BasicArgs) -> None:
        """Initialize the backend."""
        super().__init__(vectors=vectors, arguments=arguments)

    def __len__(self) -> int:
        """Get the number of vectors."""
        return self.vectors.shape[0]

    @property
    def backend_type(self) -> Backend:
        """The type of the backend."""
        return Backend.BASIC

    @abstractmethod
    def _dist(self, x: npt.NDArray) -> npt.NDArray:
        """Compute distances between x and self._vectors based on the metric."""
        raise NotImplementedError()

    @classmethod
    def from_vectors(cls, vectors: npt.NDArray, metric: Union[str, Metric] = "cosine", **kwargs: Any) -> BasicBackend:
        """Create a new instance from vectors."""
        metric_enum = Metric.from_string(metric)
        if metric_enum not in cls.supported_metrics:
            raise ValueError(f"Metric '{metric_enum.value}' is not supported by BasicBackend.")

        arguments = BasicArgs(metric=metric_enum)
        if metric_enum == Metric.COSINE:
            return CosineBasicBackend(vectors, arguments)
        elif metric_enum == Metric.EUCLIDEAN:
            return EuclideanBasicBackend(vectors, arguments)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    @classmethod
    def load(cls, folder: Path) -> BasicBackend:
        """Load the vectors from a path."""
        arguments = BasicArgs.load(folder / "arguments.json")
        vectors = cls._load_vectors(folder)
        if arguments.metric == Metric.COSINE:
            return CosineBasicBackend(vectors, arguments)
        elif arguments.metric == Metric.EUCLIDEAN:
            return EuclideanBasicBackend(vectors, arguments)
        else:
            raise ValueError(f"Unsupported metric: {arguments.metric}")

    def save(self, folder: Path) -> None:
        """Save the vectors to a path."""
        super().save(folder)
        self.arguments.dump(folder / "arguments.json")

    def threshold(
        self,
        vectors: npt.NDArray,
        threshold: float,
    ) -> list[npt.NDArray]:
        """
        Batched distance thresholding.

        :param vectors: The vectors to threshold.
        :param threshold: The threshold to use.
        :return: A list of lists of indices of vectors that are below the threshold
        """
        out: list[npt.NDArray] = []
        for i in range(0, len(vectors), 1024):
            batch = vectors[i : i + 1024]
            distances = self._dist(batch)
            for dists in distances:
                indices = np.flatnonzero(dists <= threshold)
                sorted_indices = indices[np.argsort(dists[indices])]
                out.append(sorted_indices)
        return out

    def query(
        self,
        vectors: npt.NDArray,
        k: int,
    ) -> QueryResult:
        """
        Batched distance query.

        :param vectors: The vectors to query.
        :param k: The number of nearest neighbors to return.
        :return: A list of tuples with the indices and distances.
        :raises ValueError: If k is less than 1.
        """
        if k < 1:
            raise ValueError(f"k should be >= 1, is now {k}")

        out: QueryResult = []
        num_vectors = len(self.vectors)
        effective_k = min(k, num_vectors)

        # Batch the queries
        for index in range(0, len(vectors), 1024):
            batch = vectors[index : index + 1024]
            distances = self._dist(batch)

            # Efficiently get the k smallest distances
            indices = np.argpartition(distances, kth=effective_k - 1, axis=1)[:, :effective_k]
            sorted_indices = np.take_along_axis(
                indices, np.argsort(np.take_along_axis(distances, indices, axis=1)), axis=1
            )
            sorted_distances = np.take_along_axis(distances, sorted_indices, axis=1)

            # Extend the output with tuples of (indices, distances)
            out.extend(zip(sorted_indices, sorted_distances))

        return out


class CosineBasicBackend(BasicBackend):
    def __init__(self, vectors: npt.NDArray, arguments: BasicArgs) -> None:
        """Initialize the cosine basic backend."""
        super().__init__(vectors=vectors, arguments=arguments)
        self._vectors = normalize_or_copy(self._vectors)

    def _dist(self, x: npt.NDArray) -> npt.NDArray:
        """Compute cosine distance."""
        x_norm = normalize(x)
        sim = x_norm.dot(self._vectors.T)
        return 1 - sim

    def insert(self, vectors: npt.NDArray) -> None:
        """Insert vectors into the vector space."""
        # Normalize the new vectors
        _norm_vectors = normalize_or_copy(vectors)
        self._vectors = np.vstack([self._vectors, _norm_vectors])


class EuclideanBasicBackend(BasicBackend):
    def __init__(self, vectors: npt.NDArray, arguments: BasicArgs) -> None:
        """Initialize the Euclidean basic backend."""
        super().__init__(vectors=vectors, arguments=arguments)
        self.squared_norm_vectors = (self._vectors**2).sum(1)

    def _update_precomputed_data(self) -> None:
        """Update precomputed data for Euclidean distance."""
        self.squared_norm_vectors = (self._vectors**2).sum(1)

    def _dist(self, x: npt.NDArray) -> npt.NDArray:
        """Compute Euclidean distance."""
        x_norm = (x**2).sum(1)
        dists_squared = (x_norm[:, None] + self.squared_norm_vectors[None, :]) - 2 * (x @ self._vectors.T)
        # Ensure non-negative distances
        dists_squared = np.clip(dists_squared, 0, None)
        return np.sqrt(dists_squared)
