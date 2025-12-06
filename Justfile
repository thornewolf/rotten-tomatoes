# Set project root in PYTHONPATH for all recipes
export PYTHONPATH := justfile_directory()

# Default recipe - list all available commands
default:
    @just --list

# Compile Python files to check for syntax errors
compile:
    uv run python -m compileall services app scripts

# === Development Server ===

# Start the FastAPI development server with hot reload
dev:
    uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Start the server without reload (production-like)
serve:
    uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

# === Training ===

# Train model using Tron dataset (default)
train:
    uv run python scripts/train.py --dataset tron

# Train model using dummy/synthetic data
train-dummy:
    uv run python scripts/train.py --dataset dummy --generate

# Train any configured dataset
train-dataset name:
    uv run python scripts/train.py --dataset {{name}}

# List all available datasets
list-datasets:
    uv run python scripts/train.py --list-datasets

# === Market Search & Data ===

# Search Kalshi markets for a term
search-market term="Rotten":
    uv run python scripts/search_market.py --market "{{term}}"

# Fetch market data from a Kalshi URL
fetch-market url:
    uv run python scripts/fetch_market_from_url.py "{{url}}"

# Find and save recent Rotten Tomatoes markets from Kalshi
find-rt-markets:
    uv run python scripts/find_past_rt_markets.py

# Run the scraper (placeholder)
scrape:
    uv run python scripts/scrape.py

# === Dependencies ===

# Install dependencies
install:
    uv sync

# Update dependencies
update:
    uv lock --upgrade && uv sync

# Add a new dependency
add dep:
    uv add {{dep}}

# Add a dev dependency
add-dev dep:
    uv add --dev {{dep}}

# === Code Quality ===

# Type check the codebase
typecheck:
    uv run python -m mypy app services scripts --ignore-missing-imports

# Format code with ruff
fmt:
    uv run ruff format .

# Lint code with ruff
lint:
    uv run ruff check .

# Lint and auto-fix
lint-fix:
    uv run ruff check --fix .

# === Utilities ===

# Open a Python REPL with project dependencies
repl:
    uv run python

# Start Jupyter notebook server
notebook:
    uv run jupyter notebook

# Clean Python cache files
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

# Show project structure
tree:
    tree -I '__pycache__|.git|.venv|node_modules|*.pyc' -L 3
