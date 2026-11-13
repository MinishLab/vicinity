from pathlib import Path

import numpy as np
import pytest

from vicinity import Metric, Vicinity


@pytest.mark.parametrize("metric", [Metric.COSINE, Metric.EUCLIDEAN])
@pytest.mark.parametrize("old_store,new_store", [(False, False), (False, True), (True, False), (True, True)])
def test_overwrite_preserves_current_vector_store(
    tmp_path: Path, metric: Metric, old_store: bool, new_store: bool
) -> None:
    """Overwriting a snapshot preserves only the current instance's raw vectors."""
    folder = tmp_path / "snapshot"
    old = Vicinity.from_vectors_and_items(
        np.array([[10.0, 0.0]]), ["old-document"], metric=metric, store_vectors=old_store
    )
    vectors = np.array([[0.0, 20.0]])
    current = Vicinity.from_vectors_and_items(vectors, ["new-document"], metric=metric, store_vectors=new_store)
    old.save(folder)
    current.save(folder, overwrite=True)

    loaded = Vicinity.load(folder)
    assert loaded.items == current.items
    assert loaded.query(vectors, k=1) == current.query(vectors, k=1)
    if new_store:
        assert loaded.vector_store is not None
        np.testing.assert_array_equal(loaded.get_vector_by_index(0), vectors)
    else:
        assert loaded.vector_store is None
        assert not (folder / "store").exists()
        with pytest.raises(ValueError, match="No vector store was provided"):
            loaded.get_vector_by_index(0)


@pytest.mark.parametrize("metric", [Metric.COSINE, Metric.EUCLIDEAN])
def test_overwrite_without_store_can_change_dimensions(tmp_path: Path, metric: Metric) -> None:
    """Replacing a snapshot cannot attach raw vectors with the previous dimensionality."""
    folder = tmp_path / "snapshot"
    old = Vicinity.from_vectors_and_items(np.array([[10.0, 0.0]]), ["old-document"], metric=metric, store_vectors=True)
    vectors = np.array([[0.0, 20.0, 0.0]])
    current = Vicinity.from_vectors_and_items(vectors, ["new-document"], metric=metric)
    old.save(folder)
    current.save(folder, overwrite=True)

    loaded = Vicinity.load(folder)
    assert loaded.dim == 3
    assert loaded.query(vectors, k=1) == current.query(vectors, k=1)
    assert loaded.vector_store is None


@pytest.mark.parametrize("store_vectors", [False, True])
def test_save_without_overwrite_preserves_existing_snapshot(tmp_path: Path, store_vectors: bool) -> None:
    """A rejected overwrite leaves the original persisted snapshot intact."""
    folder = tmp_path / "snapshot"
    vectors = np.array([[10.0, 0.0]])
    old = Vicinity.from_vectors_and_items(vectors, ["old-document"], store_vectors=store_vectors)
    current = Vicinity.from_vectors_and_items(np.array([[0.0, 20.0]]), ["new-document"])
    old.save(folder)
    with pytest.raises(FileExistsError):
        current.save(folder)

    loaded = Vicinity.load(folder)
    assert loaded.items == old.items
    assert loaded.query(vectors, k=1) == old.query(vectors, k=1)
    assert (loaded.vector_store is not None) == store_vectors
    if store_vectors:
        np.testing.assert_array_equal(loaded.get_vector_by_index(0), vectors)


def test_fresh_snapshot_without_raw_vectors(tmp_path: Path) -> None:
    """A fresh snapshot without stored raw vectors keeps vector lookup unavailable."""
    folder = tmp_path / "snapshot"
    vectors = np.array([[0.0, 20.0]])
    current = Vicinity.from_vectors_and_items(vectors, ["new-document"])
    current.save(folder)

    loaded = Vicinity.load(folder)
    assert loaded.query(vectors, k=1) == current.query(vectors, k=1)
    assert loaded.vector_store is None
    with pytest.raises(ValueError, match="No vector store was provided"):
        loaded.get_vector_by_index(0)


def test_overwrite_preserves_unrelated_store_files(tmp_path: Path) -> None:
    """Replacing optional raw vectors leaves unrelated files in the snapshot untouched."""
    folder = tmp_path / "snapshot"
    old = Vicinity.from_vectors_and_items(np.array([[10.0, 0.0]]), ["old-document"], store_vectors=True)
    current = Vicinity.from_vectors_and_items(np.array([[0.0, 20.0]]), ["new-document"])
    old.save(folder)
    note = folder / "store" / "notes.txt"
    note.write_text("keep this note")
    root_note = folder / "notes.txt"
    root_note.write_text("keep root note")
    current.save(folder, overwrite=True)

    loaded = Vicinity.load(folder)
    assert loaded.items == current.items
    assert loaded.vector_store is None
    assert not (folder / "store" / "vectors.npy").exists()
    assert note.read_text() == "keep this note"
    assert root_note.read_text() == "keep root note"
