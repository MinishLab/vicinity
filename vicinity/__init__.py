"""Small vector store."""

from vicinity.datatypes import Backend
from vicinity.utils import normalize
from vicinity.version import __version__
from vicinity.vicinity import Vicinity

__all__ = ["Backend", "Vicinity", "normalize", "__version__"]
