import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from huggingface_hub import DatasetCard, upload_file, upload_folder

from vicinity.backends import BasicVectorStore, get_backend_class
from vicinity.datatypes import Backend

if TYPE_CHECKING:
    from vicinity.vicinity import Vicinity


class HuggingFaceMixin:
    def save_to_hub(
        self,
        repo_id: str,
        token: str | None = None,
        private: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Save the Vicinity instance to the Hugging Face Hub.

        Args:
            repo_id: The repository ID on the Hugging Face Hub
            token: Optional authentication token for private repositories
            private: Whether to create a private repository
            **kwargs: Additional arguments passed to push_to_hub()

        """
        self.push_to_hub(repo_id, token=token, private=private, **kwargs)

    def push_to_hub(
        self,
        repo_id: str,
        token: str | None = None,
        private: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Push the Vicinity instance to the Hugging Face Hub.

        Args:
            repo_id: The repository ID on the Hugging Face Hub
            token: Optional authentication token for private repositories
            private: Whether to create a private repository
            **kwargs: Additional arguments passed to Dataset.push_to_hub()

        """
        from datasets import Dataset

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
            config = {"metadata": self.metadata, "backend_type": self.backend.backend_type.value}
            config_path = temp_path / "config.json"
            config_path.write_text(json.dumps(config))
            upload_file(
                repo_id=repo_id,
                path_or_fileobj=config_path,
                token=token,
                repo_type="dataset",
                path_in_repo="config.json",
            )

        # DatasetCard
        DatasetCard(
            content=(
                f"""
---
tags:
- vicinity
- vector-store
---

# Dataset Card for {repo_id}

This dataset was created using the [vicinity](https://github.com/MinishLab/vicinity) library, a lightweight nearest neighbors library with flexible backends.

It contains a vector space with {len(self.items)} items.

## Usage

You can load this dataset using the following code:

```python
from vicinity import Vicinity
vicinity = Vicinity.load_from_hub("{repo_id}")
```

After loading the dataset, you can use the `vicinity.query` method to find the nearest neighbors to a vector.

## Configuration

The configuration of the dataset is stored in the `config.json` file. The vector backend is stored in the `backend` folder.

```bash
{json.dumps(config, indent=2)}
```
"""
            )
        ).push_to_hub(repo_id, token=token, repo_type="dataset")

    @classmethod
    def load_from_hub(cls, repo_id: str, token: str | None = None, **kwargs: Any) -> "Vicinity":
        """
        Load a Vicinity instance from the Hugging Face Hub.

        :param repo_id: The repository ID on the Hugging Face Hub.
        :param token: Optional authentication token for private repositories.
        :param kwargs: Additional arguments passed to load_dataset.
        :return: A Vicinity instance loaded from the Hub.
        """
        from datasets import load_dataset
        from huggingface_hub import snapshot_download

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

        return cls(items=items, backend=backend, metadata=config["metadata"], vector_store=vector_store)
