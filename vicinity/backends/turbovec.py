from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy import typing as npt
from turbovec import TurboQuantIndex

from vicinity.backends.base import AbstractBackend, BaseArgs
from vicinity.datatypes import Backend, QueryResult
from vicinity.utils import Metric


@dataclass
class TurboVecArgs(BaseArgs):
    dim: int = 0
    metric: Metric = Metric.COSINE
    bit_width: int = 4


class TurboVecBackend(AbstractBackend[TurboVecArgs]):
    argument_class = TurboVecArgs
    supported_metrics = {Metric.COSINE}

    def __init__(
        self,
        index: TurboQuantIndex,
        arguments: TurboVecArgs,
    ) -> None:
        """Initialize the backend using TurboVec."""
        super().__init__(arguments)
        self.index = index

    @classmethod
    def from_vectors(
        cls: type[TurboVecBackend],
        vectors: npt.NDArray,
        metric: str | Metric = Metric.COSINE,
        bit_width: int = 4,
        **kwargs: Any,
    ) -> TurboVecBackend:
        """Create a new instance from vectors."""
        metric_enum = Metric.from_string(metric)

        if metric_enum not in cls.supported_metrics:
            raise ValueError(
                f"Metric '{metric_enum.value}' is not supported by TurboVecBackend. Only cosine is supported."
            )

        dim = vectors.shape[1]
        index = TurboQuantIndex(dim=dim, bit_width=bit_width)
        index.add(vectors.astype(np.float32))
        arguments = TurboVecArgs(dim=dim, metric=metric_enum, bit_width=bit_width)
        return cls(index, arguments)

    @property
    def backend_type(self) -> Backend:
        """The type of the backend."""
        return Backend.TURBOVEC

    @property
    def dim(self) -> int:
        """Get the dimension of the space."""
        return self.arguments.dim

    def __len__(self) -> int:
        """Get the number of vectors."""
        return len(self.index)

    @classmethod
    def load(cls: type[TurboVecBackend], path: Path) -> TurboVecBackend:
        """Load the index from a path."""
        index_path = path / "index.tq"
        arguments = TurboVecArgs.load(path / "arguments.json")
        index = TurboQuantIndex.load(str(index_path))
        return cls(index, arguments=arguments)

    def save(self, path: Path) -> None:
        """Save the index to a path."""
        self.index.write(str(path / "index.tq"))
        self.arguments.dump(path / "arguments.json")

    def query(self, vectors: npt.NDArray, k: int) -> QueryResult:
        """Query the backend and return results as tuples of keys and distances."""
        k = min(k, len(self))
        scores_batch, indices_batch = self.index.search(vectors.astype(np.float32), k=k)
        # turbovec returns cosine similarity scores (higher=better); convert to cosine distance
        distances_batch = 1.0 - scores_batch
        return [(indices_batch[i], distances_batch[i].astype(np.float32)) for i in range(len(vectors))]

    def insert(self, vectors: npt.NDArray) -> None:
        """Insert vectors into the backend."""
        self.index.add(vectors.astype(np.float32))

    def delete(self, indices: list[int]) -> None:
        """Delete vectors from the index (not supported by TurboVec)."""
        raise NotImplementedError("Dynamic deletion is not supported in TurboVec.")

    def threshold(self, vectors: npt.NDArray, threshold: float, max_k: int) -> QueryResult:
        """Query vectors within a distance threshold and return keys and distances."""
        out: QueryResult = []
        for keys_row, distances_row in self.query(vectors, max_k):
            keys_row = np.array(keys_row)
            distances_row = np.array(distances_row, dtype=np.float32)
            mask = distances_row <= threshold
            out.append((keys_row[mask], distances_row[mask]))
        return out
