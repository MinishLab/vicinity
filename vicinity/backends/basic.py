from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy import typing as npt

from vicinity.backends.base import AbstractBackend, BaseArgs
from vicinity.datatypes import Backend, Matrix, QueryResult
from vicinity.utils import normalize, normalize_or_copy


@dataclass
class BasicArgs(BaseArgs): ...


class BasicBackend(AbstractBackend[BasicArgs]):
    argument_class = BasicArgs

    def __init__(self, vectors: npt.NDArray, arguments: BasicArgs) -> None:
        """Initialize the backend using vectors."""
        super().__init__(arguments)
        self._vectors = vectors
        self._norm_vectors: npt.NDArray | None = None

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
        return cls(vectors, BasicArgs())

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
        # Make sure norm vectors is updated.
        if self._norm_vectors is not None:
            self._norm_vectors = normalize_or_copy(matrix)

    @property
    def norm_vectors(self) -> npt.NDArray:
        """
        Vectors, but normalized to unit length.

        NOTE: when all vectors are unit length, this attribute _is_ vectors.
        """
        if self._norm_vectors is None:
            self._norm_vectors = normalize_or_copy(self.vectors)
        return self._norm_vectors

    def threshold(
        self,
        vectors: npt.NDArray,
        threshold: float,
    ) -> list[npt.NDArray]:
        """Batched cosine similarity."""
        out: list[npt.NDArray] = []
        for i in range(0, len(vectors), 1024):
            batch = vectors[i : i + 1024]
            distances = self._dist(batch, self.norm_vectors)
            for _, sims in enumerate(distances):
                indices = np.flatnonzero(sims <= threshold)
                sorted_indices = indices[np.argsort(sims[indices])]
                out.append(sorted_indices)

        return out

    def query(
        self,
        vectors: npt.NDArray,
        k: int,
    ) -> QueryResult:
        """Batched cosine distance."""
        if k < 1:
            raise ValueError("num should be >= 1, is now {num}")

        out: QueryResult = []

        for index in range(0, len(vectors), 1024):
            batch = vectors[index : index + 1024]
            distances = self._dist(batch, self.norm_vectors)
            if k == 1:
                sorted_indices = np.argmin(distances, 1, keepdims=True)
            elif k >= len(self.vectors):
                # If we want more than we have, just sort everything.
                sorted_indices = np.stack([np.arange(len(self.vectors))] * len(vectors))
            else:
                sorted_indices = np.argpartition(distances, kth=k, axis=1)
                sorted_indices = sorted_indices[:, :k]
            for lidx, indices in enumerate(sorted_indices):
                dists_for_word = distances[lidx, indices]
                word_index = np.argsort(dists_for_word)
                i = indices[word_index]
                d = dists_for_word[word_index]
                out.append((i, d))

        return out

    @classmethod
    def _dist(cls, x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
        """Cosine distance function. This assumes y is normalized."""
        sim = normalize(x).dot(y.T)

        return 1 - sim

    def insert(self, vectors: npt.NDArray) -> None:
        """Insert vectors into the vector space."""
        self._vectors = np.vstack([self._vectors, vectors])

    def delete(self, indices: list[int]) -> None:
        """Deletes specific indices from the vector space."""
        self._vectors = np.delete(self._vectors, indices, axis=0)
