from __future__ import annotations

from vicinity import Vicinity
from vicinity.datatypes import Backend

BackendType = tuple[Backend, str]


def test_load_from_hub() -> None:
    """Test Vicinity.load_from_hub."""
    Vicinity.load_from_hub(repo_id="minishlab/my-vicinity-repo")
