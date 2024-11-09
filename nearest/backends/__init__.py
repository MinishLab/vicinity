from nearest.backends.base import AbstractBackend
from nearest.backends.basic import BasicBackend
from nearest.datatypes import Backend


def get_backend_class(backend: Backend | str) -> type[AbstractBackend]:
    """Get all available backends."""
    backend = Backend(backend)
    if backend == Backend.BASIC:
        return BasicBackend
    elif backend == Backend.HNSW:
        from nearest.backends.hnsw import HNSWBackend

        return HNSWBackend
    elif backend == Backend.ANNOY:
        from nearest.backends.annoy import AnnoyBackend

        return AnnoyBackend


__all__ = ["get_backend_class", "AbstractBackend"]
