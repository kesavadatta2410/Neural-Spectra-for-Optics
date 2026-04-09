"""
SpecCompress-India: Logger
Lightweight CSV + console logger. No external deps.
"""

import csv
import os
import time
from pathlib import Path
from typing import Dict, Optional


class Logger:
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir  = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / f"{experiment_name}_metrics.csv"
        self._header_written = False
        self._start_time     = time.time()

    def log(self, step: int, phase: str, metrics: Dict[str, float]) -> None:
        row = {"step": step, "phase": phase, "elapsed_s": f"{time.time() - self._start_time:.1f}"}
        row.update({k: f"{v:.6f}" for k, v in metrics.items()})

        # Console
        parts = [f"[{phase.upper()}] step={step}"]
        for k, v in metrics.items():
            parts.append(f"{k}={v:.4f}")
        print("  ".join(parts), flush=True)

        # CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)

    def info(self, msg: str) -> None:
        elapsed = time.time() - self._start_time
        print(f"[{elapsed:7.1f}s] {msg}", flush=True)
