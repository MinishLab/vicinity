from vicinity.backends.base import AbstractBackend
from vicinity.backends.basic import BasicBackend
from vicinity.datatypes import Backend


def get_backend_class(backend: Backend | str) -> type[AbstractBackend]:
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


__all__ = ["get_backend_class", "AbstractBackend"]
