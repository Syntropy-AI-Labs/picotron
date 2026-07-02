"""
Unified metrics tracking and logging curve manager.
"""

import os
import json
from typing import Dict, Any, List

class MetricsTracker:
    """Records and exports training metrics (loss, learning rate, eval parameters)."""
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.history: List[Dict[str, Any]] = []
        os.makedirs(output_dir, exist_ok=True)

    def log(self, step: int, metrics: Dict[str, Any]):
        """Record step execution metrics."""
        record = {"step": step, **metrics}
        self.history.append(record)

    def save(self):
        """Export history to JSON lines formatted files."""
        log_path = os.path.join(self.output_dir, "training_history.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4)
