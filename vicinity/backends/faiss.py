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


@dataclass
class FaissArgs(BaseArgs):
    dim: int = 0
    index_type: Literal["flat", "ivf", "hnsw", "lsh", "scalar", "pq", "ivf_scalar", "ivfpq", "ivfpqr"] = "hnsw"
    metric: Literal["cosine", "l2"] = "cosine"
    nlist: int = 100  # Used for IVF indexes
    m: int = 8  # Used for PQ and HNSW
    nbits: int = 8  # Used for LSH and PQ
    refine_nbits: int = 8  # Used for IVFPQR
    direct_map: bool = True  # Enable DirectMap for IVF indexes to allow deletion


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
        # Enable DirectMap if specified and supported by index type
        if isinstance(index, faiss.IndexIVF) and arguments.direct_map:
            index.set_direct_map_type(faiss.DirectMap.Hashtable)

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
        direct_map: bool = True,
        **kwargs: Any,
    ) -> FaissBackend:
        """Create a new instance from vectors."""
        dim = vectors.shape[1]

        # If using cosine, normalize vectors to unit length
        if metric == "cosine":
            vectors = normalize(vectors)
            faiss_metric = faiss.METRIC_INNER_PRODUCT
        else:
            faiss_metric = faiss.METRIC_L2

        if index_type == "flat":
            index = faiss.IndexFlatL2(dim) if faiss_metric == faiss.METRIC_L2 else faiss.IndexFlatIP(dim)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatL2(dim) if faiss_metric == faiss.METRIC_L2 else faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss_metric)
            index.train(vectors)
        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dim, m)
        elif index_type == "lsh":
            index = faiss.IndexLSH(dim, nbits)
        elif index_type == "scalar":
            index = faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_8bit)
            index.train(vectors)
        elif index_type == "pq":
            if not (1 <= nbits <= 16):
                logger.warning(f"Invalid nbits={nbits} for IndexPQ. Setting nbits to 16.")
                nbits = 16  # Adjust to the maximum supported value for PQ

            index = faiss.IndexPQ(dim, m, nbits)
            index.train(vectors)
        elif index_type == "ivf_scalar":
            quantizer = faiss.IndexFlatL2(dim) if faiss_metric == faiss.METRIC_L2 else faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFScalarQuantizer(quantizer, dim, nlist, faiss.ScalarQuantizer.QT_8bit)
            index.train(vectors)
        elif index_type == "ivfpq":
            quantizer = faiss.IndexFlatL2(dim) if faiss_metric == faiss.METRIC_L2 else faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
            index.train(vectors)
        elif index_type == "ivfpqr":
            quantizer = faiss.IndexFlatL2(dim) if faiss_metric == faiss.METRIC_L2 else faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFPQR(quantizer, dim, nlist, m, nbits, m, refine_nbits)
            index.train(vectors)
        else:
            raise ValueError(f"Unsupported FAISS index type: {index_type}")

        index.add(vectors)

        # Enable DirectMap for IVF indexes if requested
        if isinstance(index, faiss.IndexIVF) and direct_map:
            index.set_direct_map_type(faiss.DirectMap.Hashtable)

        arguments = FaissArgs(
            dim=dim,
            index_type=index_type,
            metric=metric,
            nlist=nlist,
            m=m,
            nbits=nbits,
            refine_nbits=refine_nbits,
            direct_map=direct_map,
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
            # Check if direct_map is enabled and use IDSelectorArray if so
            if isinstance(self.index, faiss.IndexIVF) and self.arguments.direct_map:
                id_selector = faiss.IDSelectorArray(np.array(indices, dtype=np.int64))
            else:
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

        if isinstance(
            self.index, (faiss.IndexFlat, faiss.IndexIVFFlat, faiss.IndexScalarQuantizer, faiss.IndexIVFScalarQuantizer)
        ):
            # Use range_search for supported indexes
            radius = threshold
            lims, D, I = self.index.range_search(vectors, radius)

            for i in range(vectors.shape[0]):
                start, end = lims[i], lims[i + 1]
                indices = I[start:end]
                distances = D[start:end]

                # Convert distances for cosine if needed
                if self.arguments.metric == "cosine":
                    distances = 1 - distances

                # Only include indices within the threshold
                within_threshold_indices = indices[distances < threshold]
                out.append(within_threshold_indices)
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
