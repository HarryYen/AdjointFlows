import os

from .config import ConfigManager


def load_dataset_config(adjointflows_dir, logger=None):
    """Load dataset.yaml from the adjointflows directory.

    Args:
        adjointflows_dir (str): Path to the adjointflows directory.
        logger (logging.Logger, optional): Logger for warnings/errors.

    Returns:
        dict: Parsed dataset configuration, or an empty dict if missing/invalid.
    """
    dataset_path = os.path.join(adjointflows_dir, "dataset.yaml")
    if not os.path.isfile(dataset_path):
        if logger:
            logger.warning(f"Dataset config not found: {dataset_path}")
        return {}
    try:
        cfg = ConfigManager(dataset_path)
        cfg.load()
    except Exception as exc:
        if logger:
            logger.error(f"Failed to load {dataset_path}: {exc}")
        return {}
    return cfg.config or {}


def get_by_path(data, path, default=None):
    """Get a nested value from a dict using a dot-separated path.

    Args:
        data (dict): Source dictionary.
        path (str): Dot-separated path (e.g., "seismogram.filter.P2").
        default (object): Value returned if the path is not found.

    Returns:
        object: The value at the path or the default.
    """
    if not path:
        return default
    current = data
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def deep_merge(base, override):
    """Deep-merge two dictionaries with override taking precedence."""
    if not isinstance(base, dict):
        return override
    if not isinstance(override, dict):
        return override
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
