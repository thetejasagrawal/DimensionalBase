# Contributing to DimensionalBase

Thank you for your interest in contributing to DimensionalBase!

## Development Setup

```bash
# Clone the repo
git clone https://github.com/txtgrey/DimensionalBase.git
cd DimensionalBase

# Install in development mode with all dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=dimensionalbase --cov-report=term-missing

# Run a specific test file
pytest tests/test_core.py -v
```

## Code Style

We use:
- **black** for code formatting (line length 100)
- **ruff** for linting
- **mypy** for type checking

```bash
# Format code
black dimensionalbase/ tests/

# Lint
ruff check dimensionalbase/ tests/

# Type check
mypy dimensionalbase/
```

Pre-commit hooks run these automatically on every commit.

## Commit Conventions

- Use present tense ("Add feature" not "Added feature")
- Keep the first line under 72 characters
- Reference issue numbers when applicable

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with tests
3. Ensure all tests pass: `pytest tests/ -v`
4. Ensure linting passes: `ruff check dimensionalbase/`
5. Open a PR with a clear description of the change

## Architecture Overview

```
dimensionalbase/
  db.py                  # Public API facade
  core/                  # Types and entry model
  channels/              # Storage backends (text, embedding, tensor)
  storage/               # VectorStore and migrations
  algebra/               # Dimensional operations, fingerprinting
  context/               # Budget-aware retrieval and compression
  reasoning/             # Active reasoning, confidence, provenance
  trust/                 # Agent trust model
  events/                # Pub/sub event bus
  embeddings/            # Embedding providers
  exceptions.py          # Exception hierarchy
  mcp/                   # MCP server
  server/                # REST API
  cli/                   # CLI tool
  integrations/          # LangChain, CrewAI adapters
  security/              # Auth, ACL, encryption
```

The `DimensionalBase` class in `db.py` is the single entry point that composes all subsystems. Every integration (MCP, REST, CLI, LangChain, CrewAI) wraps this class.
