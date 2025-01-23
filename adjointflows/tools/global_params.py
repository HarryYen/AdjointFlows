from pathlib import Path

def get_project_root() -> str:
    root = Path(__file__).parent.parent.parent
    if not root.exists():
        raise FileNotFoundError(f"Computed project root does not exist: {root}")
    return str(root)

GLOBAL_PARAMS = {
    "base_dir": get_project_root(),
}