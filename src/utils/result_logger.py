from __future__ import annotations

import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import torch


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _system_metadata() -> dict[str, str]:
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }


def _append_row_csv(path: str | Path, row: Mapping[str, Any], fieldnames: list[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    file_exists = p.exists()
    with p.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def log_benchmark_result(
    *,
    benchmark: str,
    params: Mapping[str, Any],
    metric_name: str,
    metric_value: float,
    out_csv: str | Path = "results/benchmarks.csv",
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    fieldnames = [
        "timestamp_utc",
        "benchmark",
        "metric_name",
        "metric_value",
        "os",
        "python",
        "torch",
        "params_json",
        "extra_json",
    ]

    row: dict[str, Any] = {
        "timestamp_utc": _utc_now_iso(),
        "benchmark": benchmark,
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        **_system_metadata(),
        "params_json": json.dumps(dict(params), sort_keys=True),
        "extra_json": json.dumps(dict(extra), sort_keys=True) if extra else "",
    }

    _append_row_csv(out_csv, row, fieldnames=fieldnames)
