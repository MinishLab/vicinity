clean:


venv:
	uv venv

install: venv
	uv sync --all-extras
	uv run pre-commit install

install-no-pre-commit:
	uv pip install ".[dev,hnsw,pynndescent,annoy,faiss,usearch]"

install-base:
	uv sync --extra dev

fix:
	uv run pre-commit run --all-files

test:
	uv run pytest --cov=vicinity --cov-report=term-missing
