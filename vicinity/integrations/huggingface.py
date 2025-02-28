from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from vicinity.backends import AbstractBackend, BasicVectorStore, get_backend_class
from vicinity.datatypes import Backend

try:
    from datasets import Dataset, load_dataset
    from huggingface_hub import CommitInfo, DatasetCard, snapshot_download, upload_file, upload_folder
except ImportError:
    raise ImportError(
        "ImportError: `datasets` and `huggingface_hub` are required to push to the Hugging Face Hub. "
        "Please install them with `pip install 'vicinity[huggingface]'`"
    )

logger = logging.getLogger(__name__)


def push_to_hub(
    repo_id: str,
    items: list[Any],
    backend: AbstractBackend,
    metadata: dict[str, str],
    vector_store: BasicVectorStore | None = None,
    model_name_or_path: str | None = None,
    token: str | None = None,
    private: bool = False,
    **kwargs: Any,
) -> "CommitInfo":
    """
    Push the Vicinity instance to the Hugging Face Hub.

    :param repo_id: The repository ID on the Hugging Face Hub.
    :param items: The items to push to the Hub.
    :param backend: The backend used to create the embeddings in the Vicinity instance.
    :param metadata: Metadata to include in the dataset card.
    :param vector_store: The vector store used to create the embeddings in the Vicinity instance.
    :param model_name_or_path: The name of the model or the path to the local directory
        that was used to create the embeddings in the Vicinity instance.
    :param token: Optional authentication token for private repositories.
    :param private: Whether to create a private repository.
    :param **kwargs: Additional arguments passed to Dataset.push_to_hub(.)
    :return: The commit info.
    """
    if isinstance(items[0], dict):
        dataset_dict = {k: [item[k] for item in items] for k in items[0].keys()}
    else:
        dataset_dict = {"items": items}
    if vector_store is not None:
        if isinstance(vector_store.vectors, np.ndarray):
            vectors: list[list[float]] = vector_store.vectors.tolist()
        dataset_dict["vectors"] = vectors

    dataset = Dataset.from_dict(dataset_dict)
    dataset.push_to_hub(repo_id, token=token, private=private, **kwargs)

    # Save backend and config files to temp directory and upload
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Save and upload backend
        backend.save(temp_path)
        upload_folder(
            repo_id=repo_id,
            folder_path=temp_path,
            token=token,
            repo_type="dataset",
            path_in_repo="backend",
        )
        # Add model_name_or_path to metadata
        if model_name_or_path is not None:
            metadata["model_name_or_path"] = model_name_or_path

        # Save and upload config
        config = {
            "metadata": metadata,
            "backend_type": backend.backend_type.value,
        }
        config_path = temp_path / "config.json"
        config_path.write_text(json.dumps(config))
        upload_file(
            repo_id=repo_id,
            path_or_fileobj=config_path,
            token=token,
            repo_type="dataset",
            path_in_repo="config.json",
        )

    # Load the dataset card template from the related path
    template_path = Path(__file__).parent / "dataset_card_template.md"
    template = template_path.read_text()
    content = template.format(repo_id=repo_id, num_items=len(items), config=json.dumps(config, indent=4))
    return DatasetCard(content=content).push_to_hub(repo_id=repo_id, token=token, repo_type="dataset")


def load_from_hub(
    repo_id: str, token: str | None = None, **kwargs: Any
) -> tuple[list[dict[str, Any]], BasicVectorStore | None, AbstractBackend, dict[str, Any]]:
    """
    Load a Vicinity instance from the Hugging Face Hub.

    :param repo_id: The repository ID on the Hugging Face Hub.
    :param token: Optional authentication token for private repositories.
    :param **kwargs: Additional arguments passed to load_dataset.
    :return: A Vicinity instance loaded from the Hub.
    """
    # Load dataset and extract items and vectors
    dataset = load_dataset(repo_id, token=token, split="train", **kwargs)
    if "items" in dataset.column_names:
        items = dataset["items"]
    else:
        # Create items from all columns except 'vectors'
        items = []
        columns = [col for col in dataset.column_names if col != "vectors"]
        for i in range(len(dataset)):
            items.append({col: dataset[col][i] for col in columns})
    has_vectors = "vectors" in dataset.column_names
    vector_store = BasicVectorStore(vectors=dataset["vectors"]) if has_vectors else None

    # Download and load config and backend
    repo_path = Path(snapshot_download(repo_id=repo_id, token=token, repo_type="dataset"))
    with open(repo_path / "config.json") as f:
        config = json.load(f)
    backend_type = Backend(config["backend_type"])
    backend = get_backend_class(backend_type).load(repo_path / "backend")
    return items, vector_store, backend, config
