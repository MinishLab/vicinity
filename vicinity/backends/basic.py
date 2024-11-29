from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy import typing as npt

from vicinity.backends.base import AbstractBackend, BaseArgs
from vicinity.datatypes import Backend, Matrix, QueryResult
from vicinity.utils import normalize, normalize_or_copy


@dataclass
class BasicArgs(BaseArgs):
    metric: Literal["cosine", "euclidean"] = "cosine"


class BasicBackend(AbstractBackend[BasicArgs]):
    argument_class = BasicArgs

    def __init__(self, vectors: npt.NDArray, arguments: BasicArgs) -> None:
        """Initialize the backend using vectors."""
        super().__init__(arguments)
        self._vectors = vectors
        self._norm_vectors: npt.NDArray | None = None
        self._squared_norm_vectors: npt.NDArray | None = None
        self._update_precomputed_data()

    def __len__(self) -> int:
        """Get the number of vectors."""
        return self.vectors.shape[0]

    @property
    def backend_type(self) -> Backend:
        """The type of the backend."""
        return Backend.BASIC

    @classmethod
    def from_vectors(cls: type[BasicBackend], vectors: npt.NDArray, **kwargs: Any) -> BasicBackend:
        """Create a new instance from vectors."""
        arguments = BasicArgs(**kwargs)
        return cls(vectors, arguments)

    @classmethod
    def load(cls: type[BasicBackend], folder: Path) -> BasicBackend:
        """Load the vectors from a path."""
        path = folder / "vectors.npy"
        arguments = BasicArgs.load(folder / "arguments.json")
        with open(path, "rb") as f:
            return cls(np.load(f), arguments)

    def save(self, folder: Path) -> None:
        """Save the vectors to a path."""
        path = Path(folder) / "vectors.npy"
        self.arguments.dump(folder / "arguments.json")
        with open(path, "wb") as f:
            np.save(f, self._vectors)

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
        matrix = np.asarray(x)
        if not np.ndim(matrix) == 2:
            raise ValueError(f"Your array does not have 2 dimensions: {np.ndim(matrix)}")
        self._vectors = matrix
        self._update_precomputed_data()

    def squared_norm(self, x: np.ndarray) -> np.ndarray:
        """Compute the squared norm of a matrix."""
        return (x**2).sum(1)

    def _update_precomputed_data(self) -> None:
        """Update precomputed data based on the metric."""
        if self.arguments.metric == "cosine":
            self._norm_vectors = normalize_or_copy(self._vectors)
        elif self.arguments.metric == "euclidean":
            self._squared_norm_vectors = self.squared_norm(self._vectors)

    @property
    def norm_vectors(self) -> npt.NDArray:
        """
        Vectors, but normalized to unit length.

        NOTE: when all vectors are unit length, this attribute _is_ vectors.
        """
        if self._norm_vectors is None:
            self._norm_vectors = normalize_or_copy(self.vectors)
        return self._norm_vectors

    @property
    def squared_norm_vectors(self) -> npt.NDArray:
        """The squared norms of the vectors."""
        if self._squared_norm_vectors is None:
            self._squared_norm_vectors = self.squared_norm(self.vectors)
        return self._squared_norm_vectors

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

            # Use argpartition for efficiency
            indices = np.argpartition(distances, kth=effective_k - 1, axis=1)[:, :effective_k]
            sorted_indices = np.take_along_axis(
                indices, np.argsort(np.take_along_axis(distances, indices, axis=1)), axis=1
            )
            sorted_distances = np.take_along_axis(distances, sorted_indices, axis=1)

            out.extend(zip(sorted_indices, sorted_distances))

        return out

    def _dist(self, x: npt.NDArray) -> npt.NDArray:
        """Compute distances between x and self._vectors based on the given metric."""
        if self.arguments.metric == "cosine":
            x_norm = normalize(x)
            sim = x_norm.dot(self.norm_vectors.T)
            return 1 - sim
        elif self.arguments.metric == "euclidean":
            x_norm = self.squared_norm(x)
            dists_squared = (x_norm[:, None] + self.squared_norm_vectors[None, :]) - 2 * (x @ self._vectors.T)

            # Ensure non-negative distances
            dists_squared = np.maximum(dists_squared, 1e-12)
            return np.sqrt(dists_squared)
        else:
            raise ValueError(f"Unsupported metric: {self.arguments.metric}")

    def insert(self, vectors: npt.NDArray) -> None:
        """Insert vectors into the vector space."""
        self._vectors = np.vstack([self._vectors, vectors])
        self._update_precomputed_data()

    def delete(self, indices: list[int]) -> None:
        """Deletes specific indices from the vector space."""
        self._vectors = np.delete(self._vectors, indices, axis=0)
        self._update_precomputed_data()
