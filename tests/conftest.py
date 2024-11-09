"""Fixtures for testing the Nearest class with pytest."""

from __future__ import annotations

import numpy as np
import pytest

from nearest import Nearest
from nearest.backends import get_backend_class
from nearest.datatypes import Backend

# Use np.random.Generator for reproducible random numbers
random_gen = np.random.default_rng(42)


@pytest.fixture(scope="session")
def items() -> list[str]:
    """Fixture providing a list of item names."""
    return [f"item{i}" for i in range(1, 101)]


@pytest.fixture(scope="session")
def vectors() -> np.ndarray:
    """Fixture providing an array of vectors corresponding to items."""
    # Generate 100 vectors of dimension 5
    return random_gen.random((100, 5))


@pytest.fixture(scope="session")
def query_vector() -> np.ndarray:
    """Fixture providing a query vector."""
    return random_gen.random(5)


@pytest.fixture(params=[Backend.BASIC, Backend.HNSW])
def backend_type(request: pytest.FixtureRequest) -> Backend:
    """Fixture parametrizing over backend types BASIC and HNSW."""
    return request.param


@pytest.fixture
def nearest_instance(backend_type: Backend, items: list[str], vectors: np.ndarray) -> Nearest:
    """Fixture creating a Nearest instance with the given backend, items, and vectors."""
    return Nearest.from_vectors_and_items(vectors, items, backend_type=backend_type)
