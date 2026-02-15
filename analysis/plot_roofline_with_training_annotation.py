from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_ceilings(path: Path) -> tuple[float, float]:
    data = json.loads(path.read_text())
    return float(data["peak_compute_gflops"]), float(data["peak_mem_bw_gbps"])


def _load_train_point(path: Path) -> tuple[float, float, str]:
    """
    Reads output from analysis/estimate_train_ai.py

    Returns:
      ai_flops_per_byte, perf_gflops, label
    """
    data = json.loads(path.read_text())
    ai = float(data["derived"]["arithmetic_intensity_flops_per_byte"])
    tflops = float(data["derived"]["tflops_est"])
    gflops = tflops * 1000.0

    inputs = data.get("inputs", {}) if isinstance(data.get("inputs", {}), dict) else {}
    dtype = str(inputs.get("dtype", "unknown"))
    bs = inputs.get("batch_size")
    seq = inputs.get("seq_len")

    label = f"train {dtype}"
    if bs is not None and seq is not None:
        label += f" (bs={bs}, seq={seq})"
    return ai, gflops, label


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot roofline and overlay one or more training points (AI + GFLOP/s).")
    p.add_argument("--ceilings", type=Path, required=True, help="Path to ceilings JSON (cpu or cuda).")
    p.add_argument(
        "--train-point",
        type=Path,
        action="append",
        required=True,
        help="Training point JSON from estimate_train_ai.py. Pass multiple times for multiple points.",
    )
    p.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    p.add_argument("--title", type=str, default="Roofline + Training Points", help="Plot title.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    peak_compute_gflops, peak_bw_gbps = _load_ceilings(args.ceilings)

    # Roofline curve
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

    # Training points
    for tp_path in args.train_point:
        train_ai, train_perf_gflops, short_label = _load_train_point(tp_path)

        plt.scatter([train_ai], [train_perf_gflops], marker="x", s=70, label=short_label)

        note = f"AI={train_ai:.2f}\nPerf={train_perf_gflops:,.0f} GFLOP/s"
        plt.annotate(note, (train_ai, train_perf_gflops), textcoords="offset points", xytext=(10, 10))

    plt.xlabel("Arithmetic Intensity (FLOP / byte)")
    plt.ylabel("Performance (GFLOP/s)")
    plt.title(args.title)
    plt.legend()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
