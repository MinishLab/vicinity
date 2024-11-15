from __future__ import annotations

import numpy as np
import pytest

from vicinity import Vicinity
from vicinity.datatypes import Backend

random_gen = np.random.default_rng(42)

# Define supported FAISS index types
FAISS_INDEX_TYPES = ["flat", "ivf", "hnsw", "lsh", "scalar", "pq", "ivf_scalar", "ivfpq", "ivfpqr"]


@pytest.fixture(scope="session")
def items() -> list[str]:
    """Fixture providing a list of item names."""
    return [f"item{i}" for i in range(1, 10001)]


@pytest.fixture(scope="session")
def vectors() -> np.ndarray:
    """Fixture providing an array of vectors sampled from a normal distribution."""
    # Sample 1000 vectors, each of dimension 8, from a normal distribution with mean=0 and std=1
    return random_gen.normal(loc=0, scale=1, size=(10000, 8))


@pytest.fixture(scope="session")
def query_vector() -> np.ndarray:
    """Fixture providing a query vector."""
    return random_gen.random(8)


BACKEND_PARAMS = [(Backend.FAISS, index_type) for index_type in FAISS_INDEX_TYPES] + [
    (Backend.BASIC, None),
    (Backend.HNSW, None),
    (Backend.ANNOY, None),
    (Backend.PYNNDESCENT, None),
]

# Create human-readable ids for each backend type
BACKEND_IDS = [f"{backend.name}-{index_type}" if index_type else backend.name for backend, index_type in BACKEND_PARAMS]


@pytest.fixture(params=BACKEND_PARAMS)
def backend_type(request: pytest.FixtureRequest) -> Backend:
    """Fixture parametrizing over all backend types defined in Backend."""
    return request.param


@pytest.fixture(params=BACKEND_PARAMS, ids=BACKEND_IDS)
def vicinity_instance(request: pytest.FixtureRequest, items: list[str], vectors: np.ndarray) -> Vicinity:
    """Fixture providing a Vicinity instance for each backend type."""
    backend_type, index_type = request.param
    # Handle FAISS backend with specific FAISS index types
    if backend_type == Backend.FAISS:
        if index_type in ("pq", "ivfpq", "ivfpqr"):
            return Vicinity.from_vectors_and_items(
                vectors, items, backend_type=backend_type, index_type=index_type, m=2, nbits=4
            )
        else:
            return Vicinity.from_vectors_and_items(
                vectors, items, backend_type=backend_type, index_type=index_type, nlist=2, nbits=32
            )

    # Handle non-FAISS backends without passing `index_type`
    return Vicinity.from_vectors_and_items(vectors, items, backend_type=backend_type)
