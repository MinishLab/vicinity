from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy import typing as npt

from nearest.backends.base import BaseBackend
from nearest.datatypes import Matrix, QueryResult
from nearest.utilities import normalize, normalize_or_copy


class BasicBackend(BaseBackend):
    def __init__(self, vectors: npt.NDArray) -> None:
        """Initialize the backend using vectors."""
        self._vectors = vectors

    def __len__(self) -> int:
        """Get the number of vectors."""
        return self.vectors.shape[0]

    @classmethod
    def load(cls: type[BasicBackend], folder: Path) -> BasicBackend:
        """Load the vectors from a path."""
        path = Path(folder) / "vectors.npy"
        with open(path, "rb") as f:
            return cls(np.load(f))

    def save(self, folder: Path) -> None:
        """Save the vectors to a path."""
        path = Path(folder) / "vectors.npy"
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
        if hasattr(self, "_norm_vectors"):
            self._norm_vectors = normalize_or_copy(matrix)

    @property
    def norm_vectors(self) -> npt.NDArray:
        """
        Vectors, but normalized to unit length.

        NOTE: when all vectors are unit length, this attribute _is_ vectors.
        """
        if not hasattr(self, "_norm_vectors"):
            self._norm_vectors = normalize_or_copy(self.vectors)
        return self._norm_vectors

    def threshold(
        self,
        vectors: npt.NDArray,
        threshold: float,
    ) -> list[list[int]]:
        """Batched cosine similarity."""
        out: list[list[int]] = []
        for i in range(0, len(vectors), 1024):
            batch = vectors[i : i + 1024]
            distances = self._dist(batch, self.norm_vectors)
            for _, sims in enumerate(distances):
                indices = np.flatnonzero(sims <= threshold)
                sorted_indices = indices[np.argsort(sims[indices])]
                out.append([d for d in sorted_indices])

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
        np.delete(self._vectors, indices, axis=0)
        # Reassign the vectors to make sure the norm vectors are updated.
        self._vectors = self._vectors
