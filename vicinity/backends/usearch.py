from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy import typing as npt
from usearch.index import Index as UsearchIndex

from vicinity.backends.base import AbstractBackend, BaseArgs
from vicinity.datatypes import Backend, QueryResult


@dataclass
class UsearchArgs(BaseArgs):
    dim: int = 0
    metric: Literal["cos", "ip", "l2sq", "hamming", "tanimoto"] = "cos"
    connectivity: int = 16
    expansion_add: int = 128
    expansion_search: int = 64


class UsearchBackend(AbstractBackend[UsearchArgs]):
    argument_class = UsearchArgs

    def __init__(
        self,
        index: UsearchIndex,
        next_key: int,
        arguments: UsearchArgs,
    ) -> None:
        """Initialize the backend using Usearch."""
        super().__init__(arguments)
        self.index = index
        self.next_key = next_key  # Keep track of the next available key

    @classmethod
    def from_vectors(
        cls: type[UsearchBackend],
        vectors: npt.NDArray,
        metric: Literal["cos", "ip", "l2sq", "hamming", "tanimoto"],
        connectivity: int,
        expansion_add: int,
        expansion_search: int,
        **kwargs: Any,
    ) -> UsearchBackend:
        """
        Create a new instance from vectors.

        :param vectors: The vectors to add to the index.
        :param metric: The metric to use for similarity search.
        :param connectivity: The connectivity parameter for the index.
        :param expansion_add: The expansion parameter for adding vectors.
        :param expansion_search: The expansion parameter for searching.
        :param **kwargs: Additional arguments.
        :return: A new UsearchBackend instance.
        """
        dim = vectors.shape[1]
        index = UsearchIndex(
            ndim=dim,
            metric=metric,
            connectivity=connectivity,
            expansion_add=expansion_add,
            expansion_search=expansion_search,
        )
        # Generate keys (IDs) for the vectors. This is needed since Usearch expects both vectors and keys.
        keys = np.arange(len(vectors))
        index.add(keys=keys, vectors=vectors)

        arguments = UsearchArgs(
            dim=dim,
            metric=metric,
            connectivity=connectivity,
            expansion_add=expansion_add,
            expansion_search=expansion_search,
        )
        next_key = len(vectors)
        return UsearchBackend(index, next_key, arguments=arguments)

    @property
    def backend_type(self) -> Backend:
        """The type of the backend."""
        return Backend.USEARCH

    @property
    def dim(self) -> int:
        """Get the dimension of the space."""
        return self.index.ndim

    def __len__(self) -> int:
        """Get the number of vectors."""
        return len(self.index)

    @classmethod
    def load(cls: type[UsearchBackend], base_path: Path) -> UsearchBackend:
        """Load the index from a path."""
        path = Path(base_path) / "index.usearch"
        arguments = UsearchArgs.load(base_path / "arguments.json")
        index = UsearchIndex(
            ndim=arguments.dim,
            metric=arguments.metric,
            connectivity=arguments.connectivity,
            expansion_add=arguments.expansion_add,
            expansion_search=arguments.expansion_search,
        )
        index.load(str(path))
        # Load next_key
        with open(base_path / "next_key.txt", "r") as f:
            next_key = int(f.read())
        return cls(index, next_key, arguments=arguments)

    def save(self, base_path: Path) -> None:
        """Save the index to a path."""
        path = Path(base_path) / "index.usearch"
        self.index.save(str(path))
        self.arguments.dump(base_path / "arguments.json")
        # Save next_key
        with open(base_path / "next_key.txt", "w") as f:
            f.write(str(self.next_key))

    def query(self, vectors: npt.NDArray, k: int) -> QueryResult:
        """Query the backend."""
        results = self.index.search(vectors, k)
        # Access the keys and distances from the results object and convert to numpy arrays
        keys = np.array(results.keys)
        distances = np.array(results.distances, dtype=np.float32)

        # If querying a single vector, reshape to (1, k)
        if keys.ndim == 1:
            keys = keys.reshape(1, -1)
            distances = distances.reshape(1, -1)

        return [(keys_row, distances_row) for keys_row, distances_row in zip(keys, distances)]

    def insert(self, vectors: npt.NDArray) -> None:
        """Insert vectors into the backend."""
        num_vectors = len(vectors)
        keys = np.arange(self.next_key, self.next_key + num_vectors)
        self.index.add(keys=keys, vectors=vectors)
        self.next_key += num_vectors

    def delete(self, keys: list[int]) -> None:
        """Delete vectors from the backend."""
        for key in keys:
            self.index.remove(key)

    def threshold(self, vectors: npt.NDArray, threshold: float) -> list[npt.NDArray]:
        """Threshold the backend."""
        out: list[npt.NDArray] = []
        for keys_row, distances_row in self.query(vectors, 100):
            keys_row = np.array(keys_row)
            distances_row = np.array(distances_row, dtype=np.float32)
            mask = distances_row < threshold
            filtered_keys = keys_row[mask]
            out.append(filtered_keys)
        return out
