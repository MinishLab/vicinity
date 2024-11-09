"""A small vector store."""

from __future__ import annotations

import logging
from io import open
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import orjson
from numpy import typing as npt

from vicinity.backends import AbstractBackend, get_backend_class
from vicinity.datatypes import Backend, PathLike

logger = logging.getLogger(__name__)


class Vicinity:
    """
    Work with vector representations of items.

    Supports functions for calculating fast batched similarity
    between items or composite representations of items.
    """

    def __init__(
        self,
        items: Sequence[str],
        backend: AbstractBackend,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize a Vicinity instance with an array and list of items.

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
        self.items: list[str] = list(items)
        self.backend: AbstractBackend = backend
        self.metadata = metadata or {}

    def __len__(self) -> int:
        """The number of the items in the vector space."""
        return len(self.items)

    @classmethod
    def from_vectors_and_items(
        cls: type[Vicinity],
        vectors: npt.NDArray,
        items: Sequence[str],
        backend_type: Backend = Backend.BASIC,
        **kwargs: Any,
    ) -> Vicinity:
        """
        Create a Vicinity instance from vectors and items.

        :param vectors: The vectors to use.
        :param items: The items to use.
        :param backend_type: The type of backend to use.
        :param **kwargs: Additional arguments to pass to the backend.
        :return: A Vicinity instance.
        """
        backend_cls = get_backend_class(backend_type)
        arguments = backend_cls.argument_class(**kwargs)
        backend = backend_cls.from_vectors(vectors, **arguments.dict())

        return cls(items, backend)

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
        :return: For each item in the input, the num most similar items are returned in the form of
            (NAME, SIMILARITY) tuples.
        """
        vectors = np.asarray(vectors)
        if np.ndim(vectors) == 1:
            vectors = vectors[None, :]

        out = []
        for index, distances in self.backend.query(vectors, k):
            distances.clip(min=0, out=distances)
            out.append([(self.items[idx], dist) for idx, dist in zip(index, distances)])

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

        :return: For each item in the input, all items above the threshold are returned.
        """
        vectors = np.array(vectors)
        if np.ndim(vectors) == 1:
            vectors = vectors[None, :]

        out = []
        for indexes in self.backend.threshold(vectors, threshold):
            out.append([self.items[idx] for idx in indexes])

        return out

    def save(
        self,
        folder: PathLike,
        overwrite: bool = False,
    ) -> None:
        """
        Save a Vicinity instance in a fast format.

        The Vicinity fast format stores the words and vectors of a Vicinity instance
        separately in a JSON and numpy format, respectively.

        :param folder: The path to which to save the JSON file. The vectors are saved separately. The JSON contains a path to the numpy file.
        :param overwrite: Whether to overwrite the JSON and numpy files if they already exist.
        :raises ValueError: If the path is not a directory.
        """
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=overwrite)

        if not path.is_dir():
            raise ValueError(f"Path {path} should be a directory.")

        items_dict = {"items": self.items, "metadata": self.metadata, "backend_type": self.backend.backend_type.value}

        with open(path / "data.json", "wb") as file_handle:
            file_handle.write(orjson.dumps(items_dict))

        self.backend.save(path)

    @classmethod
    def load(cls, filename: PathLike) -> Vicinity:
        """
        Load a Vicinity instance in fast format.

        As described above, the fast format stores the words and vectors of the
        Vicinity instance separately and is drastically faster than loading from
        .txt files.

        :param filename: The filename to load.
        :return: A Vicinity instance.
        """
        folder_path = Path(filename)

        with open(folder_path / "data.json", "rb") as file_handle:
            data: dict[str, Any] = orjson.loads(file_handle.read())
        items: Sequence[str] = data["items"]

        metadata: dict[str, Any] = data["metadata"]
        backend_type = Backend(data["backend_type"])

        backend_cls: type[AbstractBackend] = get_backend_class(backend_type)
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

        if vectors.shape[1] != self.dim:
            raise ValueError("The inserted vectors must have the same dimension as the backend.")

        item_set = set(self.items)
        for token in tokens:
            if token in item_set:
                raise ValueError(f"Token {token} is already in the vector space.")
            self.items.append(token)
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
            curr_indices = [self.items.index(token) for token in tokens]
        except KeyError as exc:
            raise ValueError(f"Token {exc} was not in the vector space.") from exc

        self.backend.delete(curr_indices)

        for index in curr_indices:
            self.items.pop(index)
