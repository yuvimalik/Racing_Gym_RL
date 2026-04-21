"""Load `.env` from the usual locations (repo root and autoresearch tree)."""

from __future__ import annotations

from pathlib import Path


def load_project_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    autoresearch_dir = Path(__file__).resolve().parent
    project_root = autoresearch_dir.parent
    load_dotenv(project_root / ".env")
    load_dotenv(autoresearch_dir / ".env")
    load_dotenv(autoresearch_dir / "results" / ".env")
