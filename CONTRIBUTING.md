# Contributing to trade-cv

Thanks for taking the time to contribute.

## Quick guidelines

- Please do not include real broker exports with personal data in issues or pull requests.
- Prefer minimal reproducible examples. If you need to share data, anonymize it.
- Keep changes focused and easy to review.

## Development setup

```bash
git clone https://github.com/MrBootes/trade_cv.git
cd trade_cv
python -m venv .venv
```

Activate:

- Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

- macOS/Linux:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -e .
```

Optional dev tooling:

```bash
pip install -r requirements-dev.txt
```

## Checks

- Compile check:

```bash
python -m compileall src
```

- Lint (optional):

```bash
ruff check .
```

## Pull requests

- Explain *why* the change is needed.
- If you add/modify a loader, include a short sample header/format description in the PR.
- Avoid reformat-only PRs unless agreed in advance.
