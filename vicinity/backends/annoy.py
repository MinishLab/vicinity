from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from annoy import AnnoyIndex
from numpy import typing as npt

from vicinity.backends.base import AbstractBackend, BaseArgs
from vicinity.datatypes import Backend, QueryResult
from vicinity.utils import normalize


@dataclass
class AnnoyArgs(BaseArgs):
    dim: int = 0
    metric: Literal["dot", "euclidean", "cosine"] = "cosine"
    trees: int = 100
    length: int | None = None


class AnnoyBackend(AbstractBackend[AnnoyArgs]):
    argument_class = AnnoyArgs

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
        metric: Literal["dot", "euclidean", "cosine"],
        trees: int,
        **kwargs: Any,
    ) -> AnnoyBackend:
        """Create a new instance from vectors."""
        dim = vectors.shape[1]
        actual_metric: Literal["dot", "euclidean"]
        if metric == "cosine":
            actual_metric = "dot"
            vectors = normalize(vectors)
        else:
            actual_metric = metric

        index = AnnoyIndex(f=dim, metric=actual_metric)
        for i, vector in enumerate(vectors):
            index.add_item(i, vector)
        index.build(trees)

        arguments = AnnoyArgs(dim=dim, trees=trees, metric=metric, length=len(vectors))
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

        metric = arguments.metric
        actual_metric = "dot" if metric == "cosine" else metric

        index = AnnoyIndex(arguments.dim, actual_metric)
        index.load(str(path))

        return cls(index, arguments=arguments)

    def save(self, base_path: Path) -> None:
        """Save the vectors to a path."""
        path = Path(base_path) / "index.bin"
        self.index.save(str(path))
        # NOTE: set the length before saving.
        self.arguments.length = len(self)
        self.arguments.dump(base_path / "arguments.json")

    def query(self, vectors: npt.NDArray, k: int) -> QueryResult:
        """Query the backend."""
        out = []
        for vec in vectors:
            if self.arguments.metric == "cosine":
                vec = normalize(vec)
            indices, scores = self.index.get_nns_by_vector(vec, k, include_distances=True)
            scores_array = np.asarray(scores)
            if self.arguments.metric == "cosine":
                # Turn cosine similarity into cosine distance.
                scores_array = 1 - scores_array
            out.append((np.asarray(indices), scores_array))
        return out

    def insert(self, vectors: npt.NDArray) -> None:
        """Insert vectors into the backend."""
        raise NotImplementedError("Annoy does not support insertion.")

    def delete(self, indices: list[int]) -> None:
        """Delete vectors from the backend."""
        raise NotImplementedError("Annoy does not support deletion.")

    def threshold(self, vectors: npt.NDArray, threshold: float) -> list[npt.NDArray]:
        """Threshold the backend."""
        out: list[npt.NDArray] = []
        for x, y in self.query(vectors, 100):
            out.append(x[y < threshold])

        return out
