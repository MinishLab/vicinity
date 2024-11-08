from nearest.backends.base import BaseBackend
from nearest.backends.basic import BasicBackend
from nearest.datatypes import Backend


def get_backend_class(backend: Backend | str) -> type[BaseBackend]:
    """Get all available backends."""
    backend = Backend(backend)
    if backend == Backend.BASIC:
        return BasicBackend
    elif backend == Backend.HNSW:
        from nearest.backends.hnsw import HnswBackend

        return HnswBackend


__all__ = ["get_backend_class", "BaseBackend"]
