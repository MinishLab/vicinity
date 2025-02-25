from __future__ import annotations

import io
import sys

from vicinity import Vicinity
from vicinity.datatypes import Backend
from vicinity.integrations.huggingface import _MODEL_NAME_OR_PATH_PRINT_STATEMENT

BackendType = tuple[Backend, str]


def test_load_from_hub(vicinity_instance: Vicinity) -> None:
    """
    Test Vicinity.load_from_hub.

    :param vicinity_instance: A Vicinity instance.
    """
    repo_id = "davidberenstein1957/my-vicinity-repo"
    # get the first part of the print statement to test if model name or path is printed
    expected_print_statement = _MODEL_NAME_OR_PATH_PRINT_STATEMENT.split(":")[0]

    # Capture the output
    captured_output = io.StringIO()
    sys.stdout = captured_output

    Vicinity.load_from_hub(repo_id=repo_id)

    # Reset redirect.
    sys.stdout = sys.__stdout__

    # Check if the expected message is in the output
    assert expected_print_statement in captured_output.getvalue()
