"""Load asr_bench/.env into os.environ. Call load_env() at the start of any entry
point that needs API keys (benchmark/run.py, app/server.py).

Kept as an explicit call (not an import side-effect) so tests don't pick up .env.
"""
from __future__ import annotations

from pathlib import Path

_ENV_PATH = Path(__file__).resolve().parent / ".env"


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(_ENV_PATH)
