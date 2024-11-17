<div align="center">

# Vicinity: The Lightweight Vector Store

</div>


<div align="center">
  <h2>
    <a href="https://pypi.org/project/vicinity/"><img src="https://img.shields.io/pypi/v/vicinity?color=%23007ec6&label=pypi%20package" alt="Package version"></a>
    <a href="https://pypi.org/project/vicinity/"><img src="https://img.shields.io/pypi/pyversions/vicinity" alt="Supported Python versions"></a>
    <a href="https://pepy.tech/project/vicinity">
    <img src="https://static.pepy.tech/badge/vicinity" alt="Downloads">
    </a>
    <a href="https://app.codecov.io/gh/MinishLab/vicinity">
    <img src="https://codecov.io/gh/MinishLab/vicinity/graph/badge.svg?token=0MQ2945OZL" alt="Codecov">
    </a>
    <a href="https://github.com/MinishLab/vicinity/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License - MIT"></a>
  </h2>
</div>


## Table of contents

- [Quickstart](#quickstart)
- [Main Features](#main-features)
- [Supported Backends](#supported-backends)
  - [Backend Parameters](#backend-parameters)
- [Usage](#usage)

Vicinity is the lightest-weight vector store. Just put in some vectors, calculate query vectors, and off you go. It provides a simple and intuitive API for nearest neighbor search, with support for different backends.

## Quickstart

Install the package with:
```bash
pip install vicinity
```

The following code snippet demonstrates how to use Vicinity for nearest neighbor search:
```python
import numpy as np
from vicinity import Vicinity
from vicinity.datatypes import Backend

# Create some dummy data
items = ["triforce", "master sword", "hylian shield", "boomerang", "hookshot"]
vectors = np.random.rand(len(items), 128)

# Initialize the Vicinity instance (using the basic backend)
vicinity = Vicinity.from_vectors_and_items(vectors=vectors, items=items, backend_type=Backend.BASIC)

# Query for nearest neighbors with a top-k search
query_vector = np.random.rand(128)
results = vicinity.query([query_vector], k=3)

# Query for nearest neighbors with a threshold search
results = vicinity.query_threshold([query_vector], threshold=0.9)

# Save the vector store
vicinity.save('my_vector_store')

# Load the vector store
vicinity = Vicinity.load('my_vector_store')
```

## Main Features
Vicinity provides the following features:
- Lightweight: Minimal dependencies and fast performance.
- Flexible Backend Support: Use different backends for vector storage and search.
- Dynamic Updates: Insert and delete items in the vector store.
- Serialization: Save and load vector stores for persistence.
- Easy to Use: Simple and intuitive API.

## Supported Backends
The following backends are supported:
- `BASIC`: A simple flat index for vector storage and search.
- `HNSW`: Hierarchical Navigable Small World Graph for approximate nearest neighbor search.
- `FAISS`: All FAISS indexes for approximate nearest neighbor search are supported.
- `ANNOY`: "Approximate Nearest Neighbors Oh Yeah" for approximate nearest neighbor search.
- `PYNNDescent`: Approximate nearest neighbor search using PyNNDescent.

### Backend Parameters

| Backend        | Parameter          | Description                                                                                   | Default Value       |
|----------------|--------------------|-----------------------------------------------------------------------------------------------|---------------------|
| **Annoy**      | `metric`           | Similarity metric to use (`dot`, `euclidean`, `cosine`).                                      | `"cosine"`          |
|                | `trees`            | Number of trees to use for indexing.                                                          | `100`               |
|                | `length`           | Optional length of the dataset.                                                               | `None`              |
| **FAISS**      | `index_type`       | Type of FAISS index (`flat`, `ivf`, `hnsw`, `lsh`, `scalar`, `pq`, `ivf_scalar`, `ivfpq`, `ivfpqr`). | `"hnsw"`           |
|                | `metric`           | Similarity metric to use (`cosine`, `l2`).                                                    | `"cosine"`          |
|                | `nlist`            | Number of cells for IVF indexes.                                                              | `100`               |
|                | `m`                | Number of subquantizers for PQ and HNSW indexes.                                              | `8`                 |
|                | `nbits`            | Number of bits for LSH and PQ indexes.                                                        | `8`                 |
|                | `refine_nbits`     | Number of bits for the refinement stage in IVFPQR indexes.                                    | `8`                 |
| **HNSW**       | `space`            | Similarity space to use (`cosine`, `l2`).                                                     | `"cosine"`          |
|                | `ef_construction`  | Size of the dynamic list during index construction.                                           | `200`               |
|                | `m`                | Number of connections per layer.                                                              | `16`                |
| **PyNNDescent**| `n_neighbors`      | Number of neighbors to use for search.                                                        | `15`                |
|                | `metric`           | Similarity metric to use (`cosine`, `euclidean`, `manhattan`).                                | `"cosine"`          |


## Usage

<details>
<summary>  Creating a Vector Store
 </summary>
<br>

You can create a Vicinity instance by providing items and their corresponding vectors:


```python
from vicinity import Vicinity
import numpy as np

items = ["triforce", "master sword", "hylian shield", "boomerang", "hookshot"]
vectors = np.random.rand(len(items), 128)

vicinity = Vicinity.from_vectors_and_items(vectors=vectors, items=items)
```

</details>

<details>
<summary>  Querying
 </summary>
<br>

Find the k nearest neighbors for a given vector:

```python
query_vector = np.random.rand(128)
results = vicinity.query([query_vector], k=3)
```

Find all neighbors within a given threshold:

```python
query_vector = np.random.rand(128)
results = vicinity.query_threshold([query_vector], threshold=0.9)
```
</details>

<details>

<summary>  Inserting and Deleting Items
 </summary>
<br>

Insert new items:

```python
new_items = ["ocarina", "bow"]
new_vectors = np.random.rand(2, 128)
vicinity.insert(new_items, new_vectors)
```

Delete items:

```python
vicinity.delete(["hookshot"])
```
</details>

<details>
<summary>  Saving and Loading
 </summary>
<br>

Save the vector store:

```python
vicinity.save('my_vector_store')
```

Load the vector store:

```python
vicinity = Vicinity.load('my_vector_store')
```
</details>

## License

MIT
