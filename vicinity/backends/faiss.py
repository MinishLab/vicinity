from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import faiss
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
    ) -> None:
        """Initialize the backend using a FAISS index."""
        super().__init__(arguments)
        self.index = index

    @classmethod
    def from_vectors(  # noqa: C901
        cls: type[FaissBackend],
        vectors: npt.NDArray,
        index_type: str = "flat",
        metric: Union[str, Metric] = "cosine",
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
        if faiss_metric == faiss.METRIC_INNER_PRODUCT:
            vectors = normalize(vectors)

        dim = vectors.shape[1]

        # Handle index creation based on index_type
        if index_type == "flat":
            index = faiss.IndexFlat(dim, faiss_metric)
        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dim, m)
        elif index_type == "lsh":
            index = faiss.IndexLSH(dim, nbits)
        elif index_type == "scalar":
            index = faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_8bit)
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
                index = faiss.IndexIVFScalarQuantizer(quantizer, dim, nlist, faiss.ScalarQuantizer.QT_8bit)
            elif index_type == "ivfpq":
                index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
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
        return cls(index=index, arguments=arguments)

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
        if self.arguments.metric == "cosine":
            vectors = normalize(vectors)
        distances, indices = self.index.search(vectors, k)
        if self.arguments.metric == "cosine":
            distances = 1 - distances
        return list(zip(indices, distances))

    def insert(self, vectors: npt.NDArray) -> None:
        """Insert vectors into the backend."""
        if self.arguments.metric == "cosine":
            vectors = normalize(vectors)
        self.index.add(vectors)

    def delete(self, indices: list[int]) -> None:
        """Delete vectors from the backend."""
        raise NotImplementedError("Deletion is not supported in FAISS backends.")

    def threshold(self, vectors: npt.NDArray, threshold: float) -> list[npt.NDArray]:
        """Query vectors within a distance threshold, using range_search if supported."""
        out: list[npt.NDArray] = []
        if self.arguments.metric == "cosine":
            vectors = normalize(vectors)

        if isinstance(self.index, RANGE_SEARCH_INDEXES):
            radius = threshold
            lims, D, I = self.index.range_search(vectors, radius)
            for i in range(vectors.shape[0]):
                start, end = lims[i], lims[i + 1]
                idx = I[start:end]
                dist = D[start:end]
                if self.arguments.metric == "cosine":
                    dist = 1 - dist
                out.append(idx[dist < threshold])
        else:
            distances, indices = self.index.search(vectors, 100)
            for dist, idx in zip(distances, indices):
                if self.arguments.metric == "cosine":
                    dist = 1 - dist
                out.append(idx[dist < threshold])

        return out

    def save(self, base_path: Path) -> None:
        """Save the FAISS index and arguments."""
        faiss.write_index(self.index, str(base_path / "index.faiss"))
        self.arguments.dump(base_path / "arguments.json")

    @classmethod
    def load(cls: type[FaissBackend], base_path: Path) -> FaissBackend:
        """Load a FAISS index and arguments."""
        arguments = FaissArgs.load(base_path / "arguments.json")
        index = faiss.read_index(str(base_path / "index.faiss"))
        return cls(index=index, arguments=arguments)
