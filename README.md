# prop-recal

Reproducible analysis pipelines.

## Setup

This project uses **uv** for environment and dependency management.

```bash
uv sync
```

## Run

```bash
uv run python scripts/run.py configs/example.yaml
```

## Development

Run tests:
```bash
uv run pytest
```

Lint / format:
```bash
uv run ruff check . --fix
uv run ruff format .
```