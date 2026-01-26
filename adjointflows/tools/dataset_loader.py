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


def resolve_dataset_list_path(base_dir, dataset_entry, list_key, subdir, default=None, required=False):
    """Resolve a dataset list file path (e.g., list.evlst) to an absolute path.

    Args:
        base_dir (str): Project base directory.
        dataset_entry (dict): Dataset configuration entry.
        list_key (str): Dot-separated key to the list file (e.g., "list.evlst").
        subdir (str): DATA subdirectory (e.g., "evlst", "stlst").
        default (str, optional): Default value if list_key is missing.
        required (bool): If True, raise when the value is missing.

    Returns:
        str or None: Absolute path to the list file, or None when missing and not required.
    """
    value = get_by_path(dataset_entry, list_key, default=default)
    if not value:
        if required:
            raise ValueError(f"Missing {list_key} in dataset config.")
        return None
    if os.path.isabs(value):
        return value
    return os.path.join(base_dir, "DATA", subdir, value)

