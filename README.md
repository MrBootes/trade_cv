# trade-cv

Repository: https://github.com/MrBootes/trade_cv

Trade-cv is a Python application for fast, practical calculation and visualization of brokerage trades.
It loads broker exports, validates/normalizes them into a consistent internal representation, computes portfolio metrics, optionally enriches instruments/prices using public market data sources, and builds an interactive HTML dashboard using Plotly (charts) and Tabulator (tables).

This README documents the exact repository layout you’re publishing (files from `to_publish/` placed directly at the repository root).

## Table of contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Run](#run)
- [Input formats](#input-formats)
- [How it works (program flow)](#how-it-works-program-flow)
- [Outputs](#outputs)
- [Repository layout](#repository-layout)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [License](#license)

## Features

- Loads and normalizes multiple trade input styles:
  - **ONES** (legacy: UNO): one-sided trade event rows
  - **TWOS** (legacy: INOUT): two-sided “round-trip” rows
  - **FLOW**: deposits/withdrawals cashflow timeline
- Data validation to prevent silent mistakes (dates, numeric parsing, required fields).
- Interactive dashboard (HTML) with Plotly figures and Tabulator tables.
- Visual/table export from the dashboard (where available).

## Requirements

- Python 3.10+

## Installation

Clone the repository and install in an isolated environment.

```bash
git clone https://github.com/MrBootes/trade_cv.git
cd trade_cv
python -m venv .venv
```

Activate the environment:

- Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

- macOS/Linux:

```bash
source .venv/bin/activate
```

Install (editable mode):

```bash
pip install -e .
```

## Run

Trade-cv is interactive: it will prompt you for inputs (format choice, files, time window, etc.).

Run via the console script:

```bash
trade-cv
```

Or run directly from source without installing (recommended for quick local runs):

```bash
python src/main.py
```

Or run as a Python module:

```bash
python -m main_workers.main_calc
```

Or (equivalent) run the package directly:

```bash
python -m main_workers
```

If you run it as a module from the repository root without installing, ensure `src/` is on `PYTHONPATH`.

- Windows (PowerShell):

```powershell
$env:PYTHONPATH = "src"
python -m main_workers.main_calc
```

- macOS/Linux:

```bash
PYTHONPATH=src python -m main_workers.main_calc
```

Note: `main_workers/main_calc.py` uses package-relative imports (e.g., `from . import starter`) for cleanliness. This means `python src/main_workers/main_calc.py` (direct file execution) is not supported; use `trade-cv`, `python src/main.py`, or `python -m main_workers.main_calc`.

## Input formats

The project supports multiple logical trade formats. Exact columns vary by broker export; the loaders validate what they need and will raise clear errors if required fields are missing or malformed.

### ONES (legacy: UNO)

“One-sided” rows: each row is a trade event (buy or sell) and the row carries the signed quantity/volume (or includes a side field that can be mapped).

Commonly present fields include:

- instrument identifier (ticker/symbol, sometimes ISIN)
- instrument type (share / bond / ETF / etc.)
- trade date/time
- price
- quantity / volume
- fee / commission (optional)
- currency (optional)
- market/provider for instrument metadata (optional)

Market/provider note (v0.1.1 and lower): public enrichment is implemented for **MOEX**.

### TWOS (legacy: INOUT)

“Two-sided” rows: a single row represents a closed position/round-trip with both legs:

- buy date + buy price
- sell date + sell price
- a single absolute quantity that applies to both legs

This format is common in broker reports that show “completed deals” as one row.

### FLOW (optional)

Cashflow timeline of deposits/withdrawals. If provided, it is used alongside trades to build a more complete cash/portfolio timeline.

## How it works (program flow)

At a high level:

1) `src/__main__.py` starts the interactive run and delegates to the main worker.
2) `src/main_workers/starter.py` handles input selection and loading.
3) `src/loaders/` loads files and validates/normalizes them:
   - `ones_load.py` – ONES/UNO loader
   - `twos_load.py` – TWOS/INOUT loader
   - `flow_load.py` – FLOW loader
   - `load_validate.py` – shared validation/normalization rules
   - `load_file.py` – file reading helpers
4) `src/main_workers/main_calc.py` performs calculations and orchestration.
5) `src/main_workers/calc_visual.py` builds the Plotly/Tabulator dashboard and writes the final HTML.
6) `src/file_managers/` contains utilities (including FIFO matching logic in `fifo_match.py`).

## Outputs

The primary output is an interactive HTML dashboard which typically includes:

- Plotly charts for portfolio dynamics and derived metrics
- Tabulator tables for detailed inspection of trades/positions

Exporting (when available) is triggered from the dashboard itself and depends on the visual/table type.

## Repository layout

This repository uses a “src layout”, but in the current published version the package directory is literally named `src`.

```
trade_cv/
  pyproject.toml
  README.md
  LICENSE
  src/
    __init__.py
    __main__.py
    loaders/
    main_workers/
    file_managers/
```

## Troubleshooting

### Import / entry point problems

Confirm what is installed in your environment:

```bash
python -c "import src; print(src.__file__)"
```

If `trade-cv` is not found, reinstall in the active environment:

```bash
pip install -e .
```

### Dashboard exports don’t download

Some browsers restrict downloads depending on how the HTML is opened.

- Prefer opening the generated HTML in a regular browser tab.
- Ensure your browser allows downloads for local files.

### Validation errors on input files

Broker exports often vary (column names, date formats, decimal separators). Trade-cv performs strict validation to avoid silent mis-computation.

If you hit a validation error:

- Confirm you selected the correct input format (ONES vs TWOS vs FLOW).
- Check that dates and numeric columns contain consistent values.
- If your broker uses different column naming, add a mapping layer in the appropriate loader.

## Development

Editable install is recommended:

```bash
pip install -e .
```

Quick compile check:

```bash
python -m compileall src
```

## Additional files

- `requirements.txt` – runtime dependencies (convenience)
- `CONTRIBUTING.md` – contribution guide
- `SECURITY.md` – vulnerability reporting policy

## License

MIT. See `LICENSE`.
