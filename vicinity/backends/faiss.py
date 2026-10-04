from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from numpy import typing as npt

from vicinity.backends.base import AbstractBackend, BaseArgs
from vicinity.datatypes import Backend, QueryResult
from vicinity.utils import Metric, normalize

logger = logging.getLogger(__name__)

# FAISS indexes that support range_search
RANGE_SEARCH_INDEXES = (
    faiss.IndexFlat,
    faiss.IndexIVFFlat,
    faiss.IndexScalarQuantizer,
    faiss.IndexIVFScalarQuantizer,
)
# FAISS indexes that need to be trained before adding vectors
TRAINABLE_INDEXES = (
    faiss.IndexIVFFlat,
    faiss.IndexScalarQuantizer,
    faiss.IndexIVFScalarQuantizer,
    faiss.IndexIVFPQ,
    faiss.IndexPQ,
    faiss.IndexIVFPQR,
)


@dataclass
class FaissArgs(BaseArgs):
    dim: int = 0
    index_type: str = "flat"
    metric: Metric = Metric.COSINE
    nlist: int = 100
    m: int = 8
    nbits: int = 8
    refine_nbits: int = 8


class FaissBackend(AbstractBackend[FaissArgs]):
    argument_class = FaissArgs
    supported_metrics = {Metric.COSINE, Metric.EUCLIDEAN}
    inverse_metric_mapping = {
        Metric.COSINE: faiss.METRIC_INNER_PRODUCT,
        Metric.EUCLIDEAN: faiss.METRIC_L2,
    }

    def __init__(
        self,
        index: faiss.Index,
        arguments: FaissArgs,
        zero_indices: npt.NDArray | None = None,
    ) -> None:
        """Initialize the backend using a FAISS index."""
        super().__init__(arguments)
        self.index = index
        # Indices of zero vectors, whose cosine distance cannot be derived from squared L2.
        self.zero_indices = np.zeros(0, dtype=np.int64) if zero_indices is None else zero_indices

    @classmethod
    def from_vectors(  # noqa: C901
        cls: type[FaissBackend],
        vectors: npt.NDArray,
        index_type: str = "flat",
        metric: str | Metric = "cosine",
        nlist: int = 100,
        m: int = 8,
        nbits: int = 8,
        refine_nbits: int = 8,
        **kwargs: Any,
    ) -> FaissBackend:
        """Create a new instance from vectors."""
        metric_enum = Metric.from_string(metric)

        if metric_enum not in cls.supported_metrics:
            raise ValueError(f"Metric '{metric_enum.value}' is not supported by FaissBackend.")

        faiss_metric = cls._map_metric_to_string(metric_enum)
        zero_indices = np.flatnonzero(np.linalg.norm(vectors, axis=1) == 0)
        if faiss_metric == faiss.METRIC_INNER_PRODUCT:
            vectors = normalize(vectors)

        dim = vectors.shape[1]

        # Handle index creation based on index_type
        if index_type == "flat":
            index = faiss.IndexFlat(dim, faiss_metric)
        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dim, m, faiss_metric)
        elif index_type == "lsh":
            index = faiss.IndexLSH(dim, nbits)
        elif index_type == "scalar":
            index = faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_8bit, faiss_metric)
        elif index_type == "pq":
            if not (1 <= nbits <= 16):
                logger.warning(f"Invalid nbits={nbits} for IndexPQ. Setting nbits to 16.")
                nbits = 16
            index = faiss.IndexPQ(dim, m, nbits)
        elif index_type.startswith("ivf"):
            quantizer = faiss.IndexFlat(dim, faiss_metric)
            if index_type == "ivf":
                index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss_metric)
            elif index_type == "ivf_scalar":
                index = faiss.IndexIVFScalarQuantizer(
                    quantizer, dim, nlist, faiss.ScalarQuantizer.QT_8bit, faiss_metric
                )
            elif index_type == "ivfpq":
                index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits, faiss_metric)
            elif index_type == "ivfpqr":
                index = faiss.IndexIVFPQR(quantizer, dim, nlist, m, nbits, m, refine_nbits)
            else:
                raise ValueError(f"Unsupported FAISS index type: {index_type}")
        else:
            raise ValueError(f"Unsupported FAISS index type: {index_type}")

        # Train the index if needed
        if isinstance(index, TRAINABLE_INDEXES):
            index.train(vectors)

        index.add(vectors)

        arguments = FaissArgs(
            dim=dim,
            index_type=index_type,
            metric=metric_enum,
            nlist=nlist,
            m=m,
            nbits=nbits,
            refine_nbits=refine_nbits,
        )
        return cls(index=index, arguments=arguments, zero_indices=zero_indices)

    def __len__(self) -> int:
        """Return the number of vectors in the index."""
        return self.index.ntotal

    @property
    def backend_type(self) -> Backend:
        """The type of the backend."""
        return Backend.FAISS

    @property
    def dim(self) -> int:
        """Get the dimension of the space."""
        return self.index.d

    def query(self, vectors: npt.NDArray, k: int) -> QueryResult:
        """Perform a k-NN search in the FAISS index."""
        k = min(len(self), k)
        zero_queries = np.linalg.norm(vectors, axis=1) == 0
        if self.arguments.metric == Metric.COSINE:
            vectors = normalize(vectors)
        distances, indices = self.index.search(vectors, k)
        out: QueryResult = []
        for idx, raw, zero_query in zip(indices, distances, zero_queries):
            # FAISS pads missing results with index -1.
            found = idx >= 0
            out.append((idx[found], self._to_distances(raw[found], idx[found], zero_query)))
        return out

    def _to_distances(self, raw: npt.NDArray, indices: npt.NDArray, zero_query: bool) -> npt.NDArray:
        """Convert raw FAISS scores to distances for the configured metric."""
        if isinstance(self.index, faiss.IndexLSH):
            # LSH returns Hamming distances between binary codes, which cannot be converted.
            return raw
        if self.index.metric_type == faiss.METRIC_INNER_PRODUCT:
            return 1 - raw
        raw = np.maximum(raw, 0)
        if self.arguments.metric != Metric.COSINE:
            return np.sqrt(raw)
        # L2 indexes (pq and ivfpqr) return squared distances, which are 2 - 2 * cosine for unit vectors.
        # Zero vectors are not unit vectors, and have cosine distance 1 to everything.
        distances = raw / 2
        distances[np.isin(indices, self.zero_indices)] = 1.0
        return np.ones_like(distances) if zero_query else distances

    def _radius(self, threshold: float) -> float:
        """Convert a distance threshold to a FAISS range search radius."""
        if self.index.metric_type == faiss.METRIC_INNER_PRODUCT:
            return 1 - threshold
        return 2 * threshold if self.arguments.metric == Metric.COSINE else threshold**2

    def insert(self, vectors: npt.NDArray) -> None:
        """Insert vectors into the backend."""
        new_zero_indices = np.flatnonzero(np.linalg.norm(vectors, axis=1) == 0) + len(self)
        self.zero_indices = np.concatenate([self.zero_indices, new_zero_indices])
        if self.arguments.metric == Metric.COSINE:
            vectors = normalize(vectors)
        self.index.add(vectors)

    def delete(self, indices: list[int]) -> None:
        """Delete vectors from the backend."""
        raise NotImplementedError("Deletion is not supported in FAISS backends.")

    def threshold(self, vectors: npt.NDArray, threshold: float, max_k: int) -> QueryResult:
        """Query vectors within a distance threshold, using range_search if supported."""
        out: QueryResult = []
        zero_queries = np.linalg.norm(vectors, axis=1) == 0
        if self.arguments.metric == Metric.COSINE:
            vectors = normalize(vectors)

        if isinstance(self.index, RANGE_SEARCH_INDEXES):
            lims, D, I = self.index.range_search(vectors, self._radius(threshold))
            results = [(I[lims[i] : lims[i + 1]], D[lims[i] : lims[i + 1]]) for i in range(vectors.shape[0])]
        else:
            distances, indices = self.index.search(vectors, max_k)
            results = list(zip(indices, distances))

        for (idx, raw), zero_query in zip(results, zero_queries):
            dist = self._to_distances(raw, idx, zero_query)
            # FAISS pads missing results with index -1.
            mask = (idx >= 0) & (dist < threshold)
            out.append((idx[mask], dist[mask]))

        return out

    def save(self, path: Path) -> None:
        """Save the FAISS index and arguments."""
        faiss.write_index(self.index, str(path / "index.faiss"))
        np.save(path / "zero_indices.npy", self.zero_indices)
        self.arguments.dump(path / "arguments.json")

    @classmethod
    def load(cls: type[FaissBackend], path: Path) -> FaissBackend:
        """Load a FAISS index and arguments."""
        arguments = FaissArgs.load(path / "arguments.json")
        index = faiss.read_index(str(path / "index.faiss"))
        zero_indices_path = path / "zero_indices.npy"
        zero_indices = np.load(zero_indices_path) if zero_indices_path.exists() else None
        return cls(index=index, arguments=arguments, zero_indices=zero_indices)
