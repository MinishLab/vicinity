<div align="center">

# Nearest: The Lightweight Vector Store

</div>

## Table of contents

- [Quickstart](#quickstart)
- [Main Features](#main-features)
- [Supported Backends](#supported-backends)
- [Usage](#usage)

Nearest is the lightest-weight vector store. Just put in some vectors, calculate query vectors, and off you go. It provides a simple and intuitive API for nearest neighbor search, with support for different backends.

## Quickstart

Install the package with:
```bash
pip install nearest
```

The following code snippet demonstrates how to use Nearest for nearest neighbor search:
```python
import numpy as np
from nearest import Nearest

# Create some dummy data
items = ["triforce", "master sword", "hylian shield", "boomerang", "hookshot"]
vectors = np.random.rand(len(items), 128)

# Initialize the Nearest instance
nearest = Nearest.from_vectors_and_items(vectors=vectors, items=items)

# Query for nearest neighbors with a top-k search
query_vector = np.random.rand(128)
results = nearest.query([query_vector], k=3)

# Query for nearest neighbors with a threshold search
results = nearest.query_threshold([query_vector], threshold=0.9)

# Save the vector store
nearest.save('my_vector_store')

# Load the vector store
nearest = Nearest.load('my_vector_store')
```

## Main Features
Nearest provides the following features:
- Lightweight: Minimal dependencies and fast performance.
- Flexible Backend Support: Use different backends for vector storage and search.
- Dynamic Updates: Insert and delete items in the vector store.
- Serialization: Save and load vector stores for persistence.
- Easy to Use: Simple and intuitive API.

## Supported Backends
The following backends are supported:
- `BASIC`: A simple flat index for vector storage and search.
- `HNSW`: Hierarchical Navigable Small World Graph for approximate nearest neighbor search.

## Usage

<details>
<summary>  Creating a Vector Store
 </summary>
<br>

You can create a Nearest instance by providing items and their corresponding vectors:


```python
from nearest import Nearest
import numpy as np

items = ["triforce", "master sword", "hylian shield", "boomerang", "hookshot"]
vectors = np.random.rand(len(items), 128)

nearest = Nearest.from_vectors_and_items(vectors=vectors, items=items)
```

</details>

<details>
<summary>  Querying
 </summary>
<br>

Find the k nearest neighbors for a given vector:

```python
query_vector = np.random.rand(128)
results = nearest.query([query_vector], k=3)
```

Find all neighbors within a given threshold:

```python
query_vector = np.random.rand(128)
results = nearest.query_threshold([query_vector], threshold=0.9)
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
nearest.insert(new_items, new_vectors)
```

Delete items:

```python
nearest.delete(["hookshot"])
```
</details>

<details>
<summary>  Saving and Loading
 </summary>
<br>

Save the vector store:

```python
nearest.save('my_vector_store')
```

Load the vector store:

```python
nearest = Nearest.load('my_vector_store')
```
</details>
