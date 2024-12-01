<div align="center">

# Vicinity: Lightweight Nearest Neighbors with Flexible Backends

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

Vicinity is a light-weight, low-dependency vector store. It provides a simple and intuitive interface for nearest neighbor search, with support for different backends and evaluation.

Instead of learning a new interface for each backend, Vicinity provides a unified interface for all backends. This allows you to easily experiment with different indexing methods and distance metrics and choose the best one for your use case. Vicinity also provides a simple way to evaluate the performance of different backends, allowing you to measure the queries per second and recall.

## Table of contents

- [Quickstart](#quickstart)
- [Main Features](#main-features)
- [Supported Backends](#supported-backends)
  - [Backend Parameters](#backend-parameters)


## Quickstart

Install the package with:
```bash
pip install vicinity
```


The following code snippet demonstrates how to use Vicinity for nearest neighbor search:
```python
import numpy as np
from vicinity import Vicinity
from vicinity.datatypes import Backend, Metric

# Create some dummy data
items = ["triforce", "master sword", "hylian shield", "boomerang", "hookshot"]
vectors = np.random.rand(len(items), 128)

# Initialize the Vicinity instance (using the basic backend and cosine metric)
vicinity = Vicinity.from_vectors_and_items(vectors=vectors, items=items, backend_type=Backend.BASIC, metric=Metric.COSINE)

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

Saving and loading a vector store:
```python
vicinity.save('my_vector_store')
vicinity = Vicinity.load('my_vector_store')
```

Evaluating a backend:
```python
# Use the first 1000 vectors as query vectors
query_vectors = vectors[:1000]

# Evaluate the Vicinity instance by measuring the queries per second and recall
qps, recall = vicinity.evaluate(
    full_vectors=vectors,
    query_vectors=query_vectors,
)
```

## Main Features
Vicinity provides the following features:
- Lightweight: Minimal dependencies and fast performance.
- Flexible Backend Support: Use different backends for vector storage and search.
- Serialization: Save and load vector stores for persistence.
- Evaluation: Easily evaluate the performance of different backends.
- Easy to Use: Simple and intuitive API.

## Supported Backends
The following backends are supported:
- `BASIC`: A simple flat index for vector storage and search.
- [HNSW](https://github.com/nmslib/hnswlib): Hierarchical Navigable Small World Graph (HNSW) for ANN search using hnswlib.
- [FAISS](https://github.com/facebookresearch/faiss): ANN search using FAISS. All FAISS indexes are supported.
- [ANNOY](https://github.com/spotify/annoy): "Approximate Nearest Neighbors Oh Yeah" for approximate nearest neighbor search.
- [PYNNDescent](https://github.com/lmcinnes/pynndescent): ANN search using PyNNDescent.
- [USEARCH](https://github.com/unum-cloud/usearch): ANN search using Usearch. This uses a highly optimized version of the HNSW algorithm.

NOTE: the ANN backends do not support dynamic deletion. To delete items, you need to recreate the index. Insertion is supported in the following backends: `FAISS`, `HNSW`, and `Usearch`. The `BASIC` backend supports both insertion and deletion.

### Backend Parameters


| Backend         | Parameter           | Description                                                                                   | Default Value       |
|-----------------|---------------------|-----------------------------------------------------------------------------------------------|---------------------|
| **Annoy**       | `metric`            | Similarity metric to use (`dot`, `euclidean`, `cosine`).                                      | `"cosine"`          |
|                 | `trees`             | Number of trees to use for indexing.                                                          | `100`               |
|                 | `length`            | Optional length of the dataset.                                                               | `None`              |
| **FAISS**       | `metric`            | Similarity metric to use (`cosine`, `l2`).                                                    | `"cosine"`          |
|                 | `index_type`        | Type of FAISS index (`flat`, `ivf`, `hnsw`, `lsh`, `scalar`, `pq`, `ivf_scalar`, `ivfpq`, `ivfpqr`). | `"hnsw"`           |
|                 | `nlist`             | Number of cells for IVF indexes.                                                              | `100`               |
|                 | `m`                 | Number of subquantizers for PQ and HNSW indexes.                                              | `8`                 |
|                 | `nbits`             | Number of bits for LSH and PQ indexes.                                                        | `8`                 |
|                 | `refine_nbits`      | Number of bits for the refinement stage in IVFPQR indexes.                                    | `8`                 |
| **HNSW**        | `metric`            | Similarity space to use (`cosine`, `l2`).                                                     | `"cosine"`          |
|                 | `ef_construction`   | Size of the dynamic list during index construction.                                           | `200`               |
|                 | `m`                 | Number of connections per layer.                                                              | `16`                |
| **PyNNDescent** | `metric`            | Similarity metric to use (`cosine`, `euclidean`, `manhattan`).                                | `"cosine"`          |
|                 | `n_neighbors`       | Number of neighbors to use for search.                                                        | `15`                |
| **Usearch**     | `metric`            | Similarity metric to use (`cos`, `ip`, `l2sq`, `hamming`, `tanimoto`).                        | `"cos"`             |
|                 | `connectivity`      | Number of connections per node in the graph.                                                  | `16`                |
|                 | `expansion_add`     | Number of candidates considered during graph construction.                                    | `128`               |
|                 | `expansion_search`  | Number of candidates considered during search.                                                | `64`                |



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
| **HNSW**       | `metric`            | Similarity space to use (`cosine`, `l2`).                                                     | `"cosine"`          |
|                | `ef_construction`  | Size of the dynamic list during index construction.                                           | `200`               |
|                | `m`                | Number of connections per layer.                                                              | `16`                |
| **PyNNDescent**| `n_neighbors`      | Number of neighbors to use for search.                                                        | `15`                |
|                | `metric`           | Similarity metric to use (`cosine`, `euclidean`, `manhattan`).                                | `"cosine"`          |


## License

MIT
