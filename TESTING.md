# Testing

Safest way to run the grader tests is to use a local virtual environment and run pytest from the repo root. This avoids global Python conflicts and makes results reproducible.

## Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pytest
python -m pytest tests/test_graders.py -v
```

## macOS / Linux (bash)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pytest
python -m pytest tests/test_graders.py -v
```

## Notes

- Run commands from the repository root.
- The grader tests do not require the full OpenEnv runtime.
