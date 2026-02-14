from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_ceilings(path: Path) -> tuple[float, float]:
    data = json.loads(path.read_text())
    peak_compute = float(data["peak_compute_gflops"])
    peak_bw = float(data["peak_mem_bw_gbps"])
    return peak_compute, peak_bw


def main() -> None:
    ceilings_path = Path("results/ceilings.json")
    peak_compute_gflops, peak_bw_gbps = _load_ceilings(ceilings_path)

    # Arithmetic intensity (AI) = FLOPs / Byte
    ai = np.logspace(-3, 3, 400)  # wide range: 1e-3 to 1e3 FLOP/byte

    # Roofline:
    # performance = min(peak_compute, peak_bw * AI)
    mem_line = peak_bw_gbps * ai
    roof = np.minimum(peak_compute_gflops, mem_line)

    # Ridge point where peak_bw*AI == peak_compute
    ridge_ai = peak_compute_gflops / peak_bw_gbps

    plt.figure()
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(ai, roof, label="Roofline")
    plt.plot(ai, mem_line, linestyle="--", label="Bandwidth limit")
    plt.axhline(peak_compute_gflops, linestyle="--", label="Compute limit")
    plt.axvline(ridge_ai, linestyle=":", label=f"Ridge AI ≈ {ridge_ai:.2f} FLOP/byte")

    plt.xlabel("Arithmetic Intensity (FLOP / byte)")
    plt.ylabel("Performance (GFLOP/s)")
    plt.title("CPU Roofline (Measured Ceilings)")
    plt.legend()

    out = Path("plots/roofline_cpu.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")

    # Also print ridge point clearly
    print(f"Ridge point AI: {ridge_ai:.4f} FLOP/byte")


if __name__ == "__main__":
    main()
