from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import pandas as pd


def _stats(df: pd.DataFrame, benchmark: str, metric_name: str) -> dict[str, float]:
    rows = df[(df["benchmark"] == benchmark) & (df["metric_name"] == metric_name)]
    if rows.empty:
        raise ValueError(f"Missing rows for benchmark={benchmark} metric_name={metric_name}")
    values = rows["metric_value"].astype(float).tolist()
    return {
        "max": float(max(values)),
        "mean": float(mean(values)),
        "n": float(len(values)),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate peak ceilings from results/benchmarks.csv.")
    p.add_argument("--csv", type=str, default="results/benchmarks.csv")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--out", type=str, default=None, help="Output JSON path (defaults by device).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.csv)

    suffix = "cpu" if args.device == "cpu" else "cuda"
    out_path = Path(args.out) if args.out else Path(f"results/ceilings_{suffix}.json")

    gemm_bench = f"peak_gemm_{suffix}"
    bw_bench = f"peak_bw_{suffix}"

    gemm = _stats(df, gemm_bench, "gflops")
    bw = _stats(df, bw_bench, "gbps")

    # We use max as the "ceiling" estimate (best observed).
    peak_compute_gflops = gemm["max"]
    peak_mem_bw_gbps = bw["max"]

    payload = {
        "device": args.device,
        "peak_compute_gflops": peak_compute_gflops,
        "peak_mem_bw_gbps": peak_mem_bw_gbps,
        "source_csv": str(args.csv),
        "gemm": gemm,
        "bw": bw,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")
    print(f"Peak compute ({args.device}): {peak_compute_gflops:.2f} GFLOP/s")
    print(f"Peak bandwidth ({args.device}): {peak_mem_bw_gbps:.2f} GB/s")


if __name__ == "__main__":
    main()
