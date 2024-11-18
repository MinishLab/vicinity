from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from numpy import typing as npt
from usearch.index import BatchMatches, Matches
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
        arguments: UsearchArgs,
    ) -> None:
        """Initialize the backend using Usearch."""
        super().__init__(arguments)
        self.index = index
        self.keys: list[Any] = []
        self.key_to_index: dict[int, int] = {}

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

        :param vectors: The vectors to index.
        :param metric: The metric to use.
        :param connectivity: The connectivity parameter.
        :param expansion_add: The expansion add parameter.
        :param expansion_search: The expansion search parameter.
        :param **kwargs: Additional keyword arguments.
        :return: A new instance of the backend.
        :raises TypeError: If the type of keys is not supported.
        """
        dim = vectors.shape[1]
        index = UsearchIndex(
            ndim=dim,
            metric=metric,
            connectivity=connectivity,
            expansion_add=expansion_add,
            expansion_search=expansion_search,
        )
        keys = index.add(keys=None, vectors=vectors)  # type: ignore
        arguments = UsearchArgs(
            dim=dim,
            metric=metric,
            connectivity=connectivity,
            expansion_add=expansion_add,
            expansion_search=expansion_search,
        )
        backend = cls(index, arguments=arguments)
        if isinstance(keys, np.ndarray):
            backend.keys = keys.tolist()
        else:
            raise TypeError(f"Unexpected type for keys: {type(keys)}")
        backend.key_to_index = {key: idx for idx, key in enumerate(backend.keys)}

        return backend

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
        return cls(index, arguments=arguments)

    def save(self, base_path: Path) -> None:
        """Save the index to a path."""
        path = Path(base_path) / "index.usearch"
        self.index.save(str(path))
        self.arguments.dump(base_path / "arguments.json")

    def query(self, vectors: npt.NDArray, k: int) -> QueryResult:
        """Query the backend."""
        results: Matches | BatchMatches = self.index.search(vectors, k)
        out: QueryResult = []

        if isinstance(results, BatchMatches):
            # Convert BatchMatches to a list
            matches_list: list[Matches] = list(results)
        else:
            # Wrap single Matches into a list
            matches_list = [results]

        for matches in matches_list:
            indices = []
            distances = []
            for key, dist in zip(matches.keys, matches.distances):
                # Map Usearch key back to Vicinity index
                idx = self.key_to_index.get(int(key))
                if idx is not None:
                    indices.append(idx)
                    distances.append(float(dist))
            out.append((np.array(indices, dtype=np.int32), np.array(distances, dtype=np.float32)))
        return out

    def insert(self, vectors: npt.NDArray) -> None:
        """Insert vectors into the backend."""
        keys: int | npt.NDArray = self.index.add(None, vectors)  # type: ignore

        # Convert single key to an array
        if isinstance(keys, int):
            keys = np.array([keys])

        start_idx = len(self.keys)
        self.keys.extend(keys.tolist())
        for i, key in enumerate(keys):
            self.key_to_index[int(key)] = start_idx + i

    def delete(self, indices: list[int]) -> None:
        """Delete vectors from the backend."""
        keys_to_delete = [self.keys[i] for i in indices]
        self.index.remove(keys_to_delete)

        # Remove keys and adjust self.keys
        for index in sorted(indices, reverse=True):
            key = self.keys[index]
            del self.keys[index]
            del self.key_to_index[key]
        # Adjust key_to_index mapping for shifted indices
        for idx, key in enumerate(self.keys):
            self.key_to_index[key] = idx

    def threshold(self, vectors: npt.NDArray, threshold: float) -> list[npt.NDArray]:
        """Threshold the backend."""
        out: list[npt.NDArray] = []
        results: Matches | BatchMatches = self.index.search(vectors, 100)

        # Ensure matches_list is always iterable
        if isinstance(results, BatchMatches):
            # Convert BatchMatches to a list
            matches_list: list[Matches] = list(results)
        else:
            # Wrap single Matches into a list
            matches_list = [results]

        for matches in matches_list:
            keys = np.array(matches.keys, dtype=np.int32)
            distances = np.array(matches.distances, dtype=np.float32)
            mask = distances < threshold
            filtered_keys = keys[mask]
            indices = []
            for key in filtered_keys:
                idx = self.key_to_index.get(int(key))
                if idx is not None:
                    indices.append(idx)
            out.append(np.array(indices, dtype=np.int32))
        return out
