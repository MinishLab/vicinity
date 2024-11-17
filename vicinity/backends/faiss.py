from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import faiss
import numpy as np
from numpy import typing as npt

from vicinity.backends.base import AbstractBackend, BaseArgs
from vicinity.datatypes import Backend, QueryResult
from vicinity.utils import normalize

logger = logging.getLogger(__name__)

# FAISS indexes that support range_search
RANGE_SEARCH_INDEXES = (faiss.IndexFlat, faiss.IndexIVFFlat, faiss.IndexScalarQuantizer, faiss.IndexIVFScalarQuantizer)
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
    index_type: Literal["flat", "ivf", "hnsw", "lsh", "scalar", "pq", "ivf_scalar", "ivfpq", "ivfpqr"] = "hnsw"
    metric: Literal["cosine", "l2"] = "cosine"
    nlist: int = 100
    m: int = 8
    nbits: int = 8
    refine_nbits: int = 8


class FaissBackend(AbstractBackend[FaissArgs]):
    argument_class = FaissArgs

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
        index_type: Literal["flat", "ivf", "hnsw", "lsh", "scalar", "pq", "ivf_scalar", "ivfpq", "ivfpqr"] = "flat",
        metric: Literal["cosine", "l2"] = "cosine",
        nlist: int = 100,
        m: int = 8,
        nbits: int = 8,
        refine_nbits: int = 8,
        **kwargs: Any,
    ) -> FaissBackend:
        """
        Create a new instance from vectors.

        :param vectors: The vectors to index.
        :param index_type: The type of FAISS index to use.
        :param metric: The metric to use for similarity search.
        :param nlist: The number of cells for IVF indexes.
        :param m: The number of subquantizers for PQ and HNSW indexes.
        :param nbits: The number of bits for LSH and PQ indexes.
        :param refine_nbits: The number of bits for the refinement stage in IVFPQR indexes.
        :param **kwargs: Additional arguments to pass to the backend.
        :return: A new FaissBackend instance.
        :raises ValueError: If an invalid index type is provided.
        """
        dim = vectors.shape[1]

        # If using cosine, normalize vectors to unit length
        if metric == "cosine":
            vectors = normalize(vectors)
            faiss_metric = faiss.METRIC_INNER_PRODUCT
        else:
            faiss_metric = faiss.METRIC_L2

        if index_type.startswith("ivf"):
            # Create a quantizer for IVF indexes
            quantizer = faiss.IndexFlatL2(dim) if faiss_metric == faiss.METRIC_L2 else faiss.IndexFlatIP(dim)

        if index_type == "flat":
            index = faiss.IndexFlatL2(dim) if faiss_metric == faiss.METRIC_L2 else faiss.IndexFlatIP(dim)
        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dim, m)
        elif index_type == "lsh":
            index = faiss.IndexLSH(dim, nbits)
        elif index_type == "scalar":
            index = faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_8bit)
        elif index_type == "pq":
            if not (1 <= nbits <= 16):
                # Log a warning and adjust nbits to the maximum supported value for PQ
                logger.warning(f"Invalid nbits={nbits} for IndexPQ. Setting nbits to 16.")
                nbits = 16
            index = faiss.IndexPQ(dim, m, nbits)
        elif index_type.startswith("ivf"):
            # Create a quantizer for IVF indexes
            quantizer = faiss.IndexFlatL2(dim) if faiss_metric == faiss.METRIC_L2 else faiss.IndexFlatIP(dim)
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

        # Train the index if needed
        if isinstance(index, TRAINABLE_INDEXES):
            index.train(vectors)

        index.add(vectors)

        # Enable DirectMap for IVF indexes so they can be used with delete
        if isinstance(index, faiss.IndexIVF):
            index.set_direct_map_type(faiss.DirectMap.Hashtable)

        arguments = FaissArgs(
            dim=dim,
            index_type=index_type,
            metric=metric,
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
        """Delete vectors from the backend, if supported."""
        if hasattr(self.index, "remove_ids"):
            if isinstance(self.index, faiss.IndexIVF):
                # Use IDSelectorArray for IVF indexes
                id_selector = faiss.IDSelectorArray(np.array(indices, dtype=np.int64))
            else:
                # Use IDSelectorBatch for other indexes
                id_selector = faiss.IDSelectorBatch(np.array(indices, dtype=np.int64))
            self.index.remove_ids(id_selector)
        else:
            raise NotImplementedError("This FAISS index type does not support deletion.")

    def threshold(self, vectors: npt.NDArray, threshold: float) -> list[npt.NDArray]:
        """Query vectors within a distance threshold, using range_search if supported."""
        out: list[npt.NDArray] = []

        # Normalize query vectors if using cosine similarity
        if self.arguments.metric == "cosine":
            vectors = normalize(vectors)

        if isinstance(self.index, RANGE_SEARCH_INDEXES):
            # Use range_search for supported indexes
            radius = threshold
            lims, D, I = self.index.range_search(vectors, radius)

            for i in range(vectors.shape[0]):
                start, end = lims[i], lims[i + 1]
                idx = I[start:end]
                dist = D[start:end]

                # Convert dist for cosine if needed
                if self.arguments.metric == "cosine":
                    dist = 1 - dist

                # Only include idx within the threshold
                within_threshold = idx[dist < threshold]
                out.append(within_threshold)
        else:
            # Fallback to search-based filtering for indexes that do not support range_search
            distances, indices = self.index.search(vectors, 100)

            for dist, idx in zip(distances, indices):
                # Convert distances for cosine if needed
                if self.arguments.metric == "cosine":
                    dist = 1 - dist
                # Filter based on the threshold
                within_threshold = idx[dist < threshold]
                out.append(within_threshold)

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
