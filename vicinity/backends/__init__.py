from typing import Union

from vicinity.backends.base import AbstractBackend
from vicinity.backends.basic import BasicBackend, BasicVectorStore
from vicinity.datatypes import Backend


def get_backend_class(backend: Union[Backend, str]) -> type[AbstractBackend]:
    """Get all available backends."""
    backend = Backend(backend)
    if backend == Backend.BASIC:
        return BasicBackend
    elif backend == Backend.HNSW:
        from vicinity.backends.hnsw import HNSWBackend

        return HNSWBackend
    elif backend == Backend.ANNOY:
        from vicinity.backends.annoy import AnnoyBackend

        return AnnoyBackend
    elif backend == Backend.PYNNDESCENT:
        from vicinity.backends.pynndescent import PyNNDescentBackend

        return PyNNDescentBackend

    elif backend == Backend.FAISS:
        from vicinity.backends.faiss import FaissBackend

        return FaissBackend

    elif backend == Backend.USEARCH:
        from vicinity.backends.usearch import UsearchBackend

        return UsearchBackend

    elif backend == Backend.VOYAGER:
        from vicinity.backends.voyager import VoyagerBackend

        return VoyagerBackend


__all__ = ["get_backend_class", "AbstractBackend", "BasicVectorStore"]
