from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_ceilings(path: Path) -> tuple[float, float]:
    data = json.loads(path.read_text())
    return float(data["peak_compute_gflops"]), float(data["peak_mem_bw_gbps"])


def _dtype_bytes(dtype: str) -> int:
    if dtype == "fp32":
        return 4
    if dtype == "fp16":
        return 2
    raise ValueError(f"Unsupported dtype: {dtype}")


def _gemm_ai(n: int, dtype: str) -> float:
    # Hand estimate: bytes ≈ (A read + B read + C write) = 3*N^2 elements
    # FLOPs ≈ 2*N^3
    bytes_moved = 3.0 * (n**2) * _dtype_bytes(dtype)
    flops = 2.0 * (n**3)
    return flops / bytes_moved


def main() -> None:
    peak_compute_gflops, peak_bw_gbps = _load_ceilings(Path("results/ceilings.json"))
    df = pd.read_csv("results/benchmarks.csv")

    # Roofline curves
    ai = np.logspace(-3, 3, 400)
    mem_line = peak_bw_gbps * ai
    roof = np.minimum(peak_compute_gflops, mem_line)
    ridge_ai = peak_compute_gflops / peak_bw_gbps

    plt.figure()
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(ai, roof, label="Roofline")
    plt.plot(ai, mem_line, linestyle="--", label="Bandwidth limit")
    plt.axhline(peak_compute_gflops, linestyle="--", label="Compute limit")
    plt.axvline(ridge_ai, linestyle=":", label=f"Ridge AI ≈ {ridge_ai:.2f} FLOP/byte")

    # Plot latest GEMM point (most recent row for peak_gemm_cpu)
    gemm_rows = df[(df["benchmark"] == "peak_gemm_cpu") & (df["metric_name"] == "gflops")]
    if not gemm_rows.empty:
        last = gemm_rows.iloc[-1]
        params = json.loads(last["params_json"])
        n = int(params["n"])
        dtype = str(params["dtype"])
        achieved = float(last["metric_value"])
        point_ai = _gemm_ai(n, dtype)

        plt.scatter([point_ai], [achieved], marker="o", s=60, label=f"GEMM (n={n}, {dtype})")
        print(f"GEMM point: AI={point_ai:.4f} FLOP/byte, achieved={achieved:.2f} GFLOP/s")

    plt.xlabel("Arithmetic Intensity (FLOP / byte)")
    plt.ylabel("Performance (GFLOP/s)")
    plt.title("CPU Roofline (Ceilings + GEMM Point)")
    plt.legend()

    out = Path("plots/roofline_cpu_with_points.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
