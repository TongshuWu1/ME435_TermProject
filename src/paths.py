from pathlib import Path

from config import OUTPUT_ROOT_DIR


def project_root() -> Path:
    """Return the project root directory (the folder containing main.py)."""
    return Path(__file__).resolve().parent.parent


def output_root() -> Path:
    out = project_root() / OUTPUT_ROOT_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_output_dir(run_label: str) -> Path:
    out = output_root() / run_label
    out.mkdir(parents=True, exist_ok=True)
    return out
