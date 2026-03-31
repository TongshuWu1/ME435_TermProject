from pathlib import Path


def project_root() -> Path:
    """Return the project root directory (the folder containing main.py)."""
    return Path(__file__).resolve().parent.parent


def output_dir(folder_name: str = 'outputs') -> Path:
    """Return the top-level outputs directory and create it if needed."""
    out = project_root() / folder_name
    out.mkdir(parents=True, exist_ok=True)
    return out
