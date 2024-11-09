"""Small vector store."""

from nearest.nearest import Nearest
from nearest.utils import normalize
from nearest.version import __version__

__all__ = ["Nearest", "normalize", "__version__"]
