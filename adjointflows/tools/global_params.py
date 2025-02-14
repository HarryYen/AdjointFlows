from pathlib import Path

def get_project_root() -> str:
    root = Path(__file__).parent.parent.parent
    if not root.exists():
        raise FileNotFoundError(f"Computed project root does not exist: {root}")
    return str(root)

GLOBAL_PARAMS = {
    "base_dir": get_project_root(),
    "mpirun_path": "/cluster/gcc630/openmpi-1.10.5/bin/mpirun",
    "mpirun_python_path": "/home/harry/.conda/envs/adjflows/bin/mpirun",
}