from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy import typing as npt
from turbovec import TurboQuantIndex

from vicinity.backends.base import AbstractBackend, BaseArgs
from vicinity.datatypes import Backend, QueryResult
from vicinity.utils import Metric, normalize

# Rows used to fit TQ+ calibration; turbovec recommends around 1024.
_CALIBRATION_SAMPLE_SIZE = 1024


@dataclass
class TurboVecArgs(BaseArgs):
    dim: int = 0
    metric: Metric = Metric.COSINE
    bit_width: int = 4
    calibrate: bool = True


class TurboVecBackend(AbstractBackend[TurboVecArgs]):
    argument_class = TurboVecArgs
    supported_metrics = {Metric.COSINE}

    def __init__(
        self,
        index: TurboQuantIndex,
        arguments: TurboVecArgs,
        positions: npt.NDArray | None = None,
    ) -> None:
        """
        Initialize the backend using TurboVec.

        :param index: The TurboVec index.
        :param arguments: The arguments of the backend.
        :param positions: The position of the vector in each index slot. Defaults to the slot itself.
        """
        super().__init__(arguments)
        self.index = index
        # Deletion moves the last vector into the freed slot, so slots are mapped back to their positions.
        self.positions = np.arange(len(index)) if positions is None else positions

    @classmethod
    def from_vectors(
        cls: type[TurboVecBackend],
        vectors: npt.NDArray,
        metric: str | Metric = Metric.COSINE,
        bit_width: int = 4,
        calibrate: bool = True,
        **kwargs: Any,
    ) -> TurboVecBackend:
        """Create a new instance from vectors, optionally fitting TQ+ calibration on a random sample first."""
        metric_enum = Metric.from_string(metric)

        if metric_enum not in cls.supported_metrics:
            raise ValueError(f"Metric '{metric_enum.value}' is not supported by TurboVecBackend.")

        if bit_width not in (2, 3, 4):
            raise ValueError(f"bit_width must be 2, 3, or 4, got {bit_width}.")

        arguments = TurboVecArgs(dim=vectors.shape[1], metric=metric_enum, bit_width=bit_width, calibrate=calibrate)
        backend = cls(TurboQuantIndex(dim=_padded_dim(arguments.dim), bit_width=bit_width), arguments)
        prepared = backend._prepare(vectors)
        # turbovec needs at least two rows to fit a calibration.
        if calibrate and len(prepared) > 1:
            rng = np.random.default_rng(42)
            sample = rng.choice(len(prepared), min(len(prepared), _CALIBRATION_SAMPLE_SIZE), replace=False)
            backend.index.calibrate(prepared[sample])
        backend.index.add(prepared)
        backend.positions = np.arange(len(prepared))
        return backend

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
        index_path = path / "index.tv"
        arguments = TurboVecArgs.load(path / "arguments.json")
        index = TurboQuantIndex.load(str(index_path))
        return cls(index, arguments=arguments, positions=np.load(path / "positions.npy"))

    def save(self, path: Path) -> None:
        """Save the index to a path."""
        self.index.write(str(path / "index.tv"))
        np.save(path / "positions.npy", self.positions)
        self.arguments.dump(path / "arguments.json")

    def query(self, vectors: npt.NDArray, k: int) -> QueryResult:
        """Query the backend and return results as tuples of keys and distances."""
        scores, indices = self.index.search(self._prepare(vectors), k=k)
        # Inner products of unit vectors are cosine similarities.
        return list(zip(self.positions[indices], 1.0 - scores))

    def insert(self, vectors: npt.NDArray) -> None:
        """Insert vectors into the backend."""
        self.positions = np.concatenate([self.positions, np.arange(len(self), len(self) + len(vectors))])
        self.index.add(self._prepare(vectors))

    def _prepare(self, vectors: npt.NDArray) -> npt.NDArray:
        """Normalize, zero-pad to the index dim and convert to contiguous float32, as turbovec requires."""
        vectors = normalize(np.asarray(vectors, dtype=np.float32))
        padding = _padded_dim(self.dim) - self.dim
        return np.ascontiguousarray(np.pad(vectors, ((0, 0), (0, padding))), dtype=np.float32)

    def delete(self, indices: list[int]) -> None:
        """Delete vectors at the given positions, shifting later positions down to stay aligned with the items."""
        deleted = np.sort(np.asarray(indices, dtype=np.int64))
        slots = np.flatnonzero(np.isin(self.positions, deleted))
        # Remove the highest slots first, so the last vector moved into a freed slot is never one being deleted.
        for slot in slots[::-1]:
            self.index.swap_remove(int(slot))
            self.positions[slot] = self.positions[-1]
            self.positions = self.positions[:-1]
        self.positions = self.positions - np.searchsorted(deleted, self.positions)

    def threshold(self, vectors: npt.NDArray, threshold: float, max_k: int) -> QueryResult:
        """Query vectors within a distance threshold and return keys and distances."""
        out: QueryResult = []
        for keys_row, distances_row in self.query(vectors, max_k):
            keys_row = np.array(keys_row)
            distances_row = np.array(distances_row, dtype=np.float32)
            mask = distances_row < threshold
            out.append((keys_row[mask], distances_row[mask]))
        return out


def _padded_dim(dim: int) -> int:
    """Round up to a multiple of 8, which turbovec requires; zero padding leaves inner products unchanged."""
    return -(-dim // 8) * 8
