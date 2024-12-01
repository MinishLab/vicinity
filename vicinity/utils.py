from __future__ import annotations

from enum import Enum
from typing import Union

import numpy as np
from numpy import typing as npt


def normalize(vectors: npt.NDArray, norms: Union[npt.NDArray, None] = None) -> npt.NDArray:
    """
    Normalize a matrix of row vectors to unit length.

    Contains a shortcut if there are no zero vectors in the matrix.
    If there are zero vectors, we do some indexing tricks to avoid
    dividing by 0.

    :param vectors: The vectors to normalize.
    :param norms: Precomputed norms. If this is None, the norms are computed.
    :return: The input vectors, normalized to unit length.
    """
    if np.ndim(vectors) == 1:
        norm_float = np.linalg.norm(vectors)
        if np.isclose(norm_float, 0):
            return np.zeros_like(vectors)
        return vectors / norm_float

    if norms is None:
        norm: npt.NDArray = np.linalg.norm(vectors, axis=1)
    else:
        norm = norms

    if np.any(np.isclose(norm, 0.0)):
        vectors = np.copy(vectors)
        nonzero = norm > 0.0
        result = np.zeros_like(vectors)
        masked_norm = norm[nonzero]
        masked_vectors = vectors[nonzero]
        result[nonzero] = masked_vectors / masked_norm[:, None]

        return result
    else:
        return vectors / norm[:, None]


def normalize_or_copy(vectors: npt.NDArray) -> npt.NDArray:
    """
    Return the original vectors if they are already normalized.

    Otherwise, the vectors are normalized, and a new array is returned.
    """
    norms = np.linalg.norm(vectors, axis=-1)
    all_unit_length = np.allclose(norms[norms != 0], 1)
    if all_unit_length:
        return vectors
    return normalize(vectors, norms)


class Metric(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    INNER_PRODUCT = "inner_product"
    L2_SQUARED = "l2sq"
    HAMMING = "hamming"
    TANIMOTO = "tanimoto"

    @classmethod
    def from_string(cls, metric: Union[str, Metric]) -> Metric:
        """Convert a string or Metric enum to a Metric enum member."""
        if isinstance(metric, cls):
            return metric
        if isinstance(metric, str):
            mapping = {
                "cos": cls.COSINE,
                "cosine": cls.COSINE,
                "dot": cls.COSINE,
                "euclidean": cls.EUCLIDEAN,
                "l2": cls.EUCLIDEAN,
                "manhattan": cls.MANHATTAN,
                "l1": cls.MANHATTAN,
                "inner_product": cls.INNER_PRODUCT,
                "ip": cls.INNER_PRODUCT,
                "l2sq": cls.L2_SQUARED,
                "l2_squared": cls.L2_SQUARED,
                "hamming": cls.HAMMING,
                "tanimoto": cls.TANIMOTO,
            }
            metric_str = metric.lower()
            if metric_str in mapping:
                return mapping[metric_str]
        raise ValueError(f"Unsupported metric: {metric}")
