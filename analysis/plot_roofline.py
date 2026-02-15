from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_ceilings(path: Path) -> tuple[float, float]:
    data = json.loads(path.read_text())
    peak_compute = float(data["peak_compute_gflops"])
    peak_bw = float(data["peak_mem_bw_gbps"])
    return peak_compute, peak_bw


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot roofline from a ceilings JSON file.")
    p.add_argument("--ceilings", type=str, default="results/ceilings_cpu.json", help="Path to ceilings JSON.")
    p.add_argument("--out", type=str, default=None, help="Output PNG path.")
    p.add_argument("--title", type=str, default=None, help="Plot title.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ceilings_path = Path(args.ceilings)

    peak_compute_gflops, peak_bw_gbps = _load_ceilings(ceilings_path)

    ai = np.logspace(-3, 3, 400)
    mem_line = peak_bw_gbps * ai
    roof = np.minimum(peak_compute_gflops, mem_line)
    ridge_ai = peak_compute_gflops / peak_bw_gbps

    title = args.title or f"Roofline ({ceilings_path.name})"
    out = Path(args.out) if args.out else Path("plots") / f"roofline_{ceilings_path.stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(ai, roof, label="Roofline")
    plt.plot(ai, mem_line, linestyle="--", label="Bandwidth limit")
    plt.axhline(peak_compute_gflops, linestyle="--", label="Compute limit")
    plt.axvline(ridge_ai, linestyle=":", label=f"Ridge AI ≈ {ridge_ai:.2f} FLOP/byte")

    plt.xlabel("Arithmetic Intensity (FLOP / byte)")
    plt.ylabel("Performance (GFLOP/s)")
    plt.title(title)
    plt.legend()

    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")
    print(f"Ridge point AI: {ridge_ai:.4f} FLOP/byte")


if __name__ == "__main__":
    main()
