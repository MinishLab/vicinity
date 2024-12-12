"""Small vector store."""

from vicinity.datatypes import Backend
from vicinity.utils import Metric, normalize
from vicinity.version import __version__
from vicinity.vicinity import Vicinity

__all__ = ["Backend", "Metric", "Vicinity", "normalize", "__version__"]
