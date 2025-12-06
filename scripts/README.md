# Training Scripts

Two training flows are available:

- Tron (prefix-to-final rating regression): predicts final % positive from early critic prefixes.
- Dummy (synthetic classification): predicts rating bucket from simulated numeric features.

## Commands

All scripts should be run from the project root using absolute imports:

- Tron pipeline (default):
  `uv run python -m scripts.train`
  - Uses `data/tron-sample.csv` → transforms to `data/processed_tron.csv` via `transform_tron.py`.
  - Model saved to `prediction_tron.model`.
  - Metrics: MAE and R².

- Dummy pipeline:
  `uv run python -m scripts.train --dataset dummy`
  - Generates `data/processed_dummy.csv` on the fly (600 rows by default).
  - Model saved to `prediction_dummy.model`.
  - Metric: accuracy.

## Other Scripts

- Search for markets:
  `uv run python -m scripts.search_market --market TICKER`

- Fetch market from URL:
  `uv run python -m scripts.fetch_market_from_url https://kalshi.com/markets/...`

## Options

- `--data-path PATH` override processed CSV output/read path.  
- `--model-path PATH` override model output path.  
- `--tron-path PATH` choose raw Tron CSV.  
- `--prefix-lengths "3,5,10,20,30,40,50"` control prefix sizes (tron only).  
- `--dummy-rows 600` control synthetic row count (dummy only).

## Notes

- **All scripts must be run from the project root** using the `-m` flag to ensure proper module imports.
- Tron labels are constant for the provided sample (one film), so metrics are not meaningful; add more titles for useful training.
- Both scripts log to stdout; no artifacts beyond processed CSVs and model files are created.
- If you're not using `uv`, replace `uv run python` with your Python interpreter (e.g., `python` or `.venv/bin/python`).
