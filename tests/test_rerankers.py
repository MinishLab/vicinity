from __future__ import annotations

import numpy as np
import pytest

from vicinity import Vicinity
from vicinity.datatypes import Backend
from vicinity.rerankers import CrossEncoderReranker


def test_reranker_initialization() -> None:
    """Test that the CrossEncoderReranker can be initialized."""
    reranker = CrossEncoderReranker()
    assert reranker.cross_encoder is not None


def test_vicinity_with_reranker(
    backend_type: Backend,
    items: list[str],
    vectors: np.ndarray,
    reranker: CrossEncoderReranker,
) -> None:
    """
    Test initializing Vicinity with a reranker.

    :param backend_type: The backend type to use,
    :param items: A list of item names.
    :param vectors: An array of vectors.
    :param reranker: A CrossEncoderReranker instance.
    """
    vicinity = Vicinity.from_vectors_and_items(
        vectors,
        items,
        backend_type=backend_type,
        reranker=reranker,
    )
    assert vicinity.reranker is not None


def test_query_with_reranker(
    vicinity_instance: Vicinity,
    query_vector: np.ndarray,
    reranker: CrossEncoderReranker,
) -> None:
    """
    Test querying Vicinity with a reranker.

    :param vicinity_instance: A Vicinity instance.
    :param query_vector: A query vector.
    :param reranker: A CrossEncoderReranker instance.
    """
    # Add a reranker to the vicinity instance
    vicinity_instance.reranker = reranker

    # Create a dummy query text
    query_text = "Test query text"
    query_vectors = query_vector[None, :]

    # Perform query with reranking
    results_with_rerank = vicinity_instance.query(
        query_vectors,
        k=5,
        query_texts=[query_text],
    )

    # Ensure results are returned
    assert len(results_with_rerank) == 1
    assert len(results_with_rerank[0]) <= 5

    # Check that scores are floats
    for item, score in results_with_rerank[0]:
        assert isinstance(score, np.float32)


def test_reranker_changes_order(
    vicinity_instance: Vicinity,
    query_vector: np.ndarray,
) -> None:
    """
    Test that the reranker changes the order of the results.

    :param vicinity_instance: A Vicinity instance.
    :param query_vector: A query vector.
    """
    # Create a dummy query text
    query_text = "Test query text"
    query_vectors = query_vector[None, :]

    # Perform query without reranking
    results_without_rerank = vicinity_instance.query(
        query_vectors,
        k=5,
    )

    # Add a reranker to the vicinity instance
    reranker = CrossEncoderReranker()
    vicinity_instance.reranker = reranker

    # Perform query with reranking
    results_with_rerank = vicinity_instance.query(
        query_vectors,
        k=5,
        query_texts=[query_text],
    )

    # Assert that the results are different
    assert results_with_rerank != results_without_rerank


def test_query_with_reranker_no_query_texts(
    vicinity_instance: Vicinity,
    query_vector: np.ndarray,
    reranker: CrossEncoderReranker,
) -> None:
    """
    Test that querying with a reranker but without query texts raises an error.

    :param vicinity_instance: A Vicinity instance.
    :param query_vector: A query vector.
    :param reranker: A CrossEncoderReranker instance.
    """
    # Add a reranker to the vicinity instance
    vicinity_instance.reranker = reranker

    with pytest.raises(ValueError):
        vicinity_instance.query(
            query_vector,
            k=5,
        )
