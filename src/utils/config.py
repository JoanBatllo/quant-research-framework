# src/utils/config.py
from pathlib import Path
from functools import lru_cache
import yaml  # Make sure to add pyyaml to requirements.txt


# Locate the settings.yaml file relative to this file
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


@lru_cache()
def get_config(path: str | Path | None = None) -> dict:
    """
    Load the project configuration from a YAML file.

    Parameters
    ----------
    path : str or Path, optional
        Custom path to a YAML config. If None, uses the default project settings.

    Returns
    -------
    dict
        Configuration as a nested dictionary.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found at: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config