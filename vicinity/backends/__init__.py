from importlib.util import find_spec
from typing import Union

from vicinity.backends.base import AbstractBackend
from vicinity.backends.basic import BasicBackend, BasicVectorStore
from vicinity.datatypes import Backend


class OptionalDependencyError(ImportError):
    def __init__(self, backend: Backend, extra: str) -> None:
        msg = f"{backend} requires extra '{extra}'.\n" f"Install it with: `pip install vicinity[{extra}]`\n"
        super().__init__(msg)
        self.backend = backend
        self.extra = extra


def _require(module_name: str, backend: Backend, extra: str) -> None:
    """Check if a dependency is importable, otherwise raise an error."""
    if find_spec(module_name) is None:
        raise OptionalDependencyError(backend, extra)


def get_backend_class(backend: Union[Backend, str]) -> type[AbstractBackend]:
    """Get all available backends."""
    backend = Backend(backend)
    if backend == Backend.BASIC:
        return BasicBackend

    elif backend == Backend.HNSW:
        _require("hnswlib", backend, "hnswlib")
        from vicinity.backends.hnsw import HNSWBackend

        return HNSWBackend
    elif backend == Backend.ANNOY:
        _require("annoy", backend, "annoy")
        from vicinity.backends.annoy import AnnoyBackend

        return AnnoyBackend
    elif backend == Backend.PYNNDESCENT:
        _require("pynndescent", backend, "pynndescent")
        from vicinity.backends.pynndescent import PyNNDescentBackend

        return PyNNDescentBackend

    elif backend == Backend.FAISS:
        _require("faiss", backend, "faiss-cpu or faiss-gpu")
        from vicinity.backends.faiss import FaissBackend

        return FaissBackend

    elif backend == Backend.USEARCH:
        _require("usearch", backend, "usearch")
        from vicinity.backends.usearch import UsearchBackend

        return UsearchBackend

    elif backend == Backend.VOYAGER:
        _require("voyager", backend, "voyager")
        from vicinity.backends.voyager import VoyagerBackend

        return VoyagerBackend


__all__ = ["get_backend_class", "AbstractBackend", "BasicVectorStore"]
