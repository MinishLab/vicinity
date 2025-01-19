from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

from numpy import typing as npt

from vicinity.backends.base import AbstractBackend, BaseArgs
from vicinity.datatypes import Backend, QueryResult
from vicinity.utils import Metric

import flatnav
from flatnav.data_type import DataType
from flatnav.index import IndexL2Float, IndexL2Int8, IndexL2Uint8
from flatnav.index import IndexIPFloat, IndexIPInt8, IndexIPUint8

INDEX_CLASS_MAPPING = {
    "l2": {
        DataType.float32: IndexL2Float,
        DataType.int8: IndexL2Int8,
        DataType.uint8: IndexL2Uint8,
    },
    "angular": {
        DataType.float32: IndexIPFloat,
        DataType.int8: IndexIPInt8,
        DataType.uint8: IndexIPUint8,
    },
}


def _get_index_class(metric: str, data_type: str):
    try:
        return INDEX_CLASS_MAPPING[metric][data_type]
    except KeyError:
        raise ValueError(
            f"Invalid metric ('{metric}') or data type ('{data_type}') specified."
        )


@dataclass
class FlatNavArgs(BaseArgs):
    # Define index construction parameters.
    metric: Metric = Metric.L2_SQUARED
    max_edges_per_node: int = 32
    ef_construction: int = 100
    num_build_threads: int = 2
    index_data_type: DataType = DataType.float32
    dim: int = 128


class FlatNavBackend(AbstractBackend[FlatNavArgs]):
    argument_class = FlatNavArgs
    supported_metrics = {Metric.L2_SQUARED}
    _metric_mapping = {Metric.L2_SQUARED: "l2"}

    def __init__(self, index, arguments: FlatNavArgs):
        super().__init__(arguments)
        self.index = index

    @classmethod
    def from_vectors(
        cls: type[FlatNavBackend],
        vectors,
        metric: Union[str, Metric],
        max_edges_per_node: int,
        ef_construction: int,
        num_build_threads: int,
        index_data_type: DataType,
        **kwargs: Any,
    ) -> FlatNavBackend:
        metric_enum = Metric.from_string(metric)

        if metric_enum not in cls.supported_metrics:
            raise ValueError(
                f"Metric '{metric_enum.value}' is not supported by FlatNavBackend."
            )

        space = cls._metric_mapping[metric_enum]

        dim = vectors.shape[1]
        size = len(vectors)
        index = flatnav.index.create(
            distance_type=space,
            index_data_type=DataType.float32,
            dim=dim,
            dataset_size=size,
            max_edges_per_node=max_edges_per_node,
            verbose=True,
            collect_stats=True,
        )
        index.set_num_threads(num_build_threads)
        index.add(data=vectors, ef_construction=ef_construction)

        return cls(
            index,
            arguments=FlatNavArgs(
                metric=metric,
                max_edges_per_node=max_edges_per_node,
                ef_construction=ef_construction,
                num_build_threads=num_build_threads,
                index_data_type=index_data_type,
                dim=dim,
            ),
        )

    def query(self, query: npt.NDArray, k: int) -> QueryResult:
        """Query the backend for the nearest neighbors."""
        k = min(k, len(self))
        distances, indices = self.index.search(
            queries=query,
            ef_search=self.arguments.ef_construction,
            K=k,
            num_initializations=100,
        )
        return list(zip(indices, distances))

    def threshold(
        self,
        vectors: npt.NDArray,
        threshold: float,
        max_k: int,
    ) -> QueryResult:
        out: list[tuple[npt.NDArray, npt.NDArray]] = []
        """Threshold the backend."""
        for x, y in self.query(vectors, max_k):
            mask = y < threshold
            out.append((x[mask], y[mask]))

        return out

    def insert(self, vectors):
        self.index.add(data=vectors, ef_construction=self.arguments.ef_construction)

    @classmethod
    def load(cls: type[FlatNavBackend], base_path: Path) -> FlatNavBackend:
        """Load the vectors from a path."""
        path = base_path + "/index.bin"
        arguments = FlatNavArgs.load(base_path / "arguments.json")

        indexclass = _get_index_class(arguments.metric, arguments.index_data_type)
        index = indexclass.load_index(str(path))
        return cls(index, arguments=arguments)

    def save(self, base_path: Path) -> None:
        """Save the vectors to a path."""
        path = Path(base_path) / "index.bin"
        self.index.save(str(path))
        self.arguments.dump(base_path / "arguments.json")

    @property
    def backend_type(self) -> Backend:
        """The type of the backend"""
        return Backend.FLATNAV

    @property
    def dim(self):
        return self.arguments.dim

    def __len__(self) -> int:
        return len(self.index.get_graph_outdegree_table())

    def delete(self, indices: list[int]) -> None:
        """Delete vectors from the backend."""
        raise NotImplementedError("Deletion is not supported in FlatNav backend.")
