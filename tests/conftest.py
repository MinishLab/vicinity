"""Fixtures for testing the Nearest class with pytest."""

from __future__ import annotations

import numpy as np
import pytest

from nearest.backends import get_backend_class
from nearest.datatypes import Backend
from nearest.nearest import Nearest


@pytest.fixture(scope="session")
def items() -> list[str]:
    """Fixture providing a list of item names."""
    return ["item1", "item2", "item3"]


@pytest.fixture(scope="session")
def vectors() -> np.ndarray:
    """Fixture providing an array of vectors corresponding to items."""
    return np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.7071, 0.7071],
        ]
    )


@pytest.fixture(params=[Backend.BASIC, Backend.HNSW])
def backend_type(request) -> Backend:
    """Fixture parametrizing over backend types BASIC and HNSW."""
    return request.param


@pytest.fixture
def nearest_instance(backend_type: Backend, items: list[str], vectors: np.ndarray) -> Nearest:
    """Fixture creating a Nearest instance with the given backend, items, and vectors."""
    return Nearest.from_vectors_and_items(vectors, items, backend_type=backend_type)
