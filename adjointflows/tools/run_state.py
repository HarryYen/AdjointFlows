import datetime
import json
import os


class RunStateManager:
    """Manage workflow state and per-iteration history files."""

    def __init__(self, base_dir, run_id=None):
        self.base_dir = base_dir
        self.state_dir = os.path.join(self.base_dir, "TOMO", ".state")
        self.history_dir = os.path.join(self.state_dir, "history")
        os.makedirs(self.history_dir, exist_ok=True)

        if run_id is None:
            run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id
        self.state_path = os.path.join(self.state_dir, "run_state.json")
        self.state = {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "status": "INIT",
            "updated_at": self._now_iso(),
        }

    def _now_iso(self):
        return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds")

    def _write_json(self, path, payload):
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def update(self, **fields):
        self.state.update(fields)
        self.state["updated_at"] = self._now_iso()
        self._write_json(self.state_path, self.state)
        return dict(self.state)

    def record_iteration(self, iter_index, attempt, payload):
        entry = {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "iter_index": int(iter_index),
            "attempt": int(attempt),
            "timestamp": self._now_iso(),
        }
        if payload:
            entry.update(payload)
        file_name = f"iter_{int(iter_index):03d}_attempt_{int(attempt):02d}.json"
        output_path = os.path.join(self.history_dir, file_name)
        self._write_json(output_path, entry)
        return output_path
