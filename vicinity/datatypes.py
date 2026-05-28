from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import TypeVar

from numpy import typing as npt

T = TypeVar("T")

PathLike = str | Path
Matrix = npt.NDArray | list[npt.NDArray]
SimilarityItem = list[tuple[T, float]]
SimilarityResult = list[list[tuple[T, float]]]
# Tuple of (indices, distances)
SingleQueryResult = tuple[npt.NDArray, npt.NDArray]
QueryResult = list[SingleQueryResult]
Tokens = Iterable[str]


class Backend(str, Enum):
    HNSW = "hnsw"
    BASIC = "basic"
    ANNOY = "annoy"
    PYNNDESCENT = "pynndescent"
    FAISS = "faiss"
    USEARCH = "usearch"
    VOYAGER = "voyager"
    TURBOVEC = "turbovec"
