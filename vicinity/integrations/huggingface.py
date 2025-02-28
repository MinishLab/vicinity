from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vicinity.backends import BasicVectorStore, get_backend_class
from vicinity.datatypes import Backend

if TYPE_CHECKING:
    from huggingface_hub import CommitInfo

    from vicinity.vicinity import Vicinity

_HUB_IMPORT_ERROR = ImportError(
    "`datasets` and `huggingface_hub` are required to push to the Hugging Face Hub. Please install them with `pip install 'vicinity[huggingface]'`"
)
_MODEL_NAME_OR_PATH_PRINT_STATEMENT = (
    "Embeddings in Vicinity instance were created from model name or path: {model_name_or_path}"
)

logger = logging.getLogger(__name__)


class HuggingFaceMixin:
    def push_to_hub(
        self,
        model_name_or_path: str,
        repo_id: str,
        token: str | None = None,
        private: bool = False,
        **kwargs: Any,
    ) -> "CommitInfo":
        """
        Push the Vicinity instance to the Hugging Face Hub.

        :param model_name_or_path: The name of the model or the path to the local directory
            that was used to create the embeddings in the Vicinity instance.
        :param repo_id: The repository ID on the Hugging Face Hub
        :param token: Optional authentication token for private repositories
        :param private: Whether to create a private repository
        :param **kwargs: Additional arguments passed to Dataset.push_to_hub()
        :return: The commit info
        """
        try:
            from datasets import Dataset
            from huggingface_hub import DatasetCard, upload_file, upload_folder
        except ImportError:
            raise _HUB_IMPORT_ERROR

        # Create and push dataset with items and vectors
        if isinstance(self.items[0], dict):
            dataset_dict = {k: [item[k] for item in self.items] for k in self.items[0].keys()}
        else:
            dataset_dict = {"items": self.items}
        if self.vector_store is not None:
            dataset_dict["vectors"] = self.vector_store.vectors
        dataset = Dataset.from_dict(dataset_dict)
        dataset.push_to_hub(repo_id, token=token, private=private, **kwargs)

        # Save backend and config files to temp directory and upload
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save and upload backend
            self.backend.save(temp_path)
            upload_folder(
                repo_id=repo_id,
                folder_path=temp_path,
                token=token,
                repo_type="dataset",
                path_in_repo="backend",
            )

            # Save and upload config
            config = {
                "metadata": self.metadata,
                "backend_type": self.backend.backend_type.value,
                "model_name_or_path": model_name_or_path,
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
        content = template.format(repo_id=repo_id, num_items=len(self.items), config=json.dumps(config, indent=4))
        return DatasetCard(content=content).push_to_hub(repo_id=repo_id, token=token, repo_type="dataset")

    @classmethod
    def load_from_hub(cls, repo_id: str, token: str | None = None, **kwargs: Any) -> "Vicinity":
        """
        Load a Vicinity instance from the Hugging Face Hub.

        :param repo_id: The repository ID on the Hugging Face Hub.
        :param token: Optional authentication token for private repositories.
        :param **kwargs: Additional arguments passed to load_dataset.
        :return: A Vicinity instance loaded from the Hub.
        """
        try:
            from datasets import load_dataset
            from huggingface_hub import snapshot_download
        except ImportError:
            raise _HUB_IMPORT_ERROR

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
            model_name_or_path = config.pop("model_name_or_path")

        print(_MODEL_NAME_OR_PATH_PRINT_STATEMENT.format(model_name_or_path=model_name_or_path))
        backend_type = Backend(config["backend_type"])
        backend = get_backend_class(backend_type).load(repo_path / "backend")

        return cls(items=items, backend=backend, metadata=config["metadata"], vector_store=vector_store)
