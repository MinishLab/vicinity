---
tags:
- vicinity
- vector-store
---

# Dataset Card for {repo_id}

This dataset was created using the [vicinity](https://github.com/MinishLab/vicinity) library, a lightweight nearest neighbors library with flexible backends.

It contains a vector space with {num_items} items.

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
{config}
```