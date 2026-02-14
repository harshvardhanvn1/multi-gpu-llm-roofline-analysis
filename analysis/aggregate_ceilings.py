from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No rows found in {path}")
    return df


def _stats(df: pd.DataFrame, benchmark: str, metric_name: str) -> dict[str, float]:
    sub = df[(df["benchmark"] == benchmark) & (df["metric_name"] == metric_name)]
    if sub.empty:
        raise ValueError(f"Missing rows for benchmark={benchmark} metric_name={metric_name}")
    vals = sub["metric_value"].astype(float)
    return {
        "count": float(len(vals)),
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "max": float(vals.max()),
        "min": float(vals.min()),
    }


def main() -> None:
    in_csv = Path("results/benchmarks.csv")
    out_json = Path("results/ceilings.json")

    df = _load_csv(in_csv)

    gemm = _stats(df, "peak_gemm_cpu", "gflops")
    bw = _stats(df, "peak_bw_cpu", "gbps")

    ceilings: dict[str, Any] = {
        "source_csv": str(in_csv),
        "peak_compute_gflops": gemm["max"],
        "peak_mem_bw_gbps": bw["max"],
        "stats": {
            "peak_gemm_cpu_gflops": gemm,
            "peak_bw_cpu_gbps": bw,
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(ceilings, indent=2, sort_keys=True))

    print(f"Wrote {out_json}")
    print(f"Peak compute (CPU): {ceilings['peak_compute_gflops']:.2f} GFLOP/s")
    print(f"Peak bandwidth (CPU): {ceilings['peak_mem_bw_gbps']:.2f} GB/s")
    print(f"GEMM samples: n={int(gemm['count'])} mean={gemm['mean']:.2f} std={gemm['std']:.2f} max={gemm['max']:.2f}")
    print(f"BW samples:   n={int(bw['count'])} mean={bw['mean']:.2f} std={bw['std']:.2f} max={bw['max']:.2f}")


if __name__ == "__main__":
    main()
