from enum import Enum
from pathlib import Path
from typing import Iterable, List, Tuple, Union

from numpy import typing as npt

PathLike = Union[str, Path]
Matrix = Union[npt.NDArray, List[npt.NDArray]]
SimilarityItem = List[Tuple[str, float]]
SimilarityResult = List[SimilarityItem]
# Tuple of (indices, distances)
SingleQueryResult = Tuple[npt.NDArray, npt.NDArray]
QueryResult = List[SingleQueryResult]
Tokens = Iterable[str]


class Backend(str, Enum):
    HNSW = "hnsw"
    BASIC = "basic"
    ANNOY = "annoy"
    PYNNDESCENT = "pynndescent"
    FAISS = "faiss"
    USEARCH = "usearch"
    VOYAGER = "voyager"
