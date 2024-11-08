"""A small vector store."""

from __future__ import annotations

import json
import logging
from io import open
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy import typing as npt

from nearest.backends import BaseBackend, get_backend_class
from nearest.datatypes import Backend, Dtype, PathLike

logger = logging.getLogger(__name__)


class Nearest:
    """
    Work with vector representations of items.

    Supports functions for calculating fast batched similarity
    between items or composite representations of items.
    """

    def __init__(
        self,
        items: Sequence[str],
        backend: BaseBackend,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize a Nearest instance with an array and list of items.

        :param items: The items in the vector space.
            A list of items. Length must be equal to the number of vectors, and
            aligned with the vectors.
        :param backend: The backend to use for the vector space.
        :param metadata: A dictionary containing metadata about the vector space.
        :raises ValueError: If the length of the items and vectors are not the same.
        """
        if len(items) != len(backend):
            raise ValueError(
                "Your vector space and list of items are not the same length: " f"{len(backend)} != {len(items)}"
            )
        if isinstance(items, (dict, set)):
            raise ValueError(
                "Your item list is a set or dict, and might not "
                "retain order in the conversion to internal look"
                "-ups. Please convert it to list and check the "
                "order."
            )
        self._items: dict[str, int] = {w: idx for idx, w in enumerate(items)}
        self._indices: dict[int, str] = {idx: w for w, idx in self.items.items()}
        self.backend: BaseBackend = backend
        self.metadata = metadata or {}

    def __len__(self) -> int:
        """The number of the items in the vector space."""
        return len(self.items)

    @classmethod
    def from_vectors_and_items(
        cls: type[Nearest],
        vectors: npt.NDArray,
        items: Sequence[str],
        backend_type: Backend = Backend.BASIC,
        **kwargs: Any,
    ) -> Nearest:
        """
        Create a Nearest instance from vectors and items.

        :param vectors: The vectors to use.
        :param items: The items to use.
        :param backend_type: The type of backend to use.
        :param **kwargs: Additional arguments to pass to the backend.
        :return: A Nearest instance.
        """
        backend_cls = get_backend_class(backend_type)
        arguments = backend_cls.argument_class(dim=vectors.shape[1], **kwargs)
        backend = backend_cls.from_vectors(vectors, **arguments.dict())
        return cls(items, backend)

    @property
    def items(self) -> dict[str, int]:
        """A mapping from item ids to their indices."""
        return self._items

    @property
    def indices(self) -> dict[int, str]:
        """A mapping from integers to item indices."""
        return self._indices

    @property
    def sorted_items(self) -> list[str]:
        """The items, sorted by index."""
        items: list[str] = [item for item, _ in sorted(self.items.items(), key=lambda x: x[1])]
        return items

    @property
    def dim(self) -> int:
        """The dimensionality of the vectors."""
        return self.backend.dim

    def query(
        self,
        vectors: npt.NDArray,
        k: int = 10,
    ) -> list[list[tuple[str, float]]]:
        """
        Find the nearest neighbors to some arbitrary vector.

        Use this to look up the nearest neighbors to a vector that is not in the vocabulary.

        :param vectors: The vectors to find the nearest neighbors to.
        :param k: The number of most similar items to retrieve.
        :return: For each item in the input, the num most similar items are returned as integers.
        """
        vectors = np.asarray(vectors)
        if np.ndim(vectors) == 1:
            vectors = vectors[None, :]

        out = []
        for index, distances in self.backend.query(vectors, k):
            distances.clip(min=0, out=distances)
            out.append([(self.indices[idx], dist) for idx, dist in zip(index, distances)])

        return out

    def query_threshold(
        self,
        vectors: npt.NDArray,
        threshold: float = 0.5,
    ) -> list[list[str]]:
        """
        Find the nearest neighbors to some arbitrary vector with some threshold.

        :param vectors: The vectors to find the most similar vectors to.
        :param threshold: The threshold to use.

        :return: For each items in the input the num most similar items are returned in the form of
            (NAME, SIMILARITY) tuples.
        """
        vectors = np.array(vectors)
        if np.ndim(vectors) == 1:
            vectors = vectors[None, :]

        out = []
        for indexes in self.backend.threshold(vectors, threshold):
            out.append([self.indices[idx] for idx in indexes])

        return out

    def save(
        self,
        folder: PathLike,
        overwrite: bool = False,
    ) -> None:
        """
        Save a nearest instance in a fast format.

        The nearest fast format stores the words and vectors of a Nearest instance
        separately in a JSON and numpy format, respectively.

        :param folder: The path to which to save the JSON file. The vectors are saved separately. The JSON contains a path to the numpy file.
        :param overwrite: Whether to overwrite the JSON and numpy files if they already exist.
        :raises ValueError: If the path is not a directory.
        """
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=overwrite)

        if not path.is_dir():
            raise ValueError(f"Path {path} should be a directory.")

        items = self.sorted_items
        items_dict = {"items": items, "metadata": self.metadata, "backend_type": self.backend.backend_type.value}

        with open(path / "data.json", "w") as file_handle:
            json.dump(items_dict, file_handle)

        self.backend.save(path)

    @classmethod
    def load(cls, filename: PathLike) -> Nearest:
        """
        Load a nearest instance in fast format.

        As described above, the fast format stores the words and vectors of the
        Nearest instance separately, and is drastically faster than loading from
        .txt files.

        :param filename: The filename to load.
        :return: A Nearest instance.
        """
        folder_path = Path(filename)

        with open(folder_path / "data.json") as file_handle:
            data: dict[str, Any] = json.load(file_handle)
        items: list[str] = data["items"]

        metadata: dict[str, Any] = data["metadata"]
        backend_type = Backend(data["backend_type"])

        backend_cls: type[BaseBackend] = get_backend_class(backend_type)
        backend = backend_cls.load(folder_path)

        instance = cls(items, backend, metadata=metadata)

        return instance

    def insert(self, tokens: Sequence[str], vectors: npt.NDArray) -> None:
        """
        Insert new items into the vector space.

        :param tokens: A list of items to insert into the vector space.
        :param vectors: The vectors to insert into the vector space.
        :raises ValueError: If the tokens and vectors are not the same length.
        """
        if len(tokens) != len(vectors):
            raise ValueError(f"Your tokens and vectors are not the same length: {len(tokens)} != {len(vectors)}")

        for token in tokens:
            if token in self.items:
                raise ValueError(f"Token {token} is already in the vector space.")
            self.items[token] = len(self.items)
            self.indices[len(self.items) - 1] = token
        self.backend.insert(vectors)

    def delete(self, tokens: Sequence[str]) -> None:
        """
        Delete tokens from the vector space.

        The removal of tokens is done in place. If the tokens are not in the vector space,
        a ValueError is raised.

        :param tokens: A list of tokens to remove from the vector space.
        :raises ValueError: If any passed tokens are not in the vector space.
        """
        try:
            curr_indices = [self.items[token] for token in tokens]
        except KeyError as exc:
            raise ValueError(f"Token {exc} was not in the vector space.") from exc

        self.backend.delete(curr_indices)

        tokens_set = set(tokens)
        new_items: dict[str, int] = {}
        for item in self.items:
            if item in tokens_set:
                tokens_set.remove(item)
                continue
            new_items[item] = len(new_items)

        self._items = new_items
        self._indices = {idx: item for item, idx in self.items.items()}
