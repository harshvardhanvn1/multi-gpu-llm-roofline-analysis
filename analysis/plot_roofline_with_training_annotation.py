from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_ceilings(path: Path) -> tuple[float, float]:
    data = json.loads(path.read_text())
    return float(data["peak_compute_gflops"]), float(data["peak_mem_bw_gbps"])


def main() -> None:
    peak_compute_gflops, peak_bw_gbps = _load_ceilings(Path("results/ceilings.json"))
    df = pd.read_csv("results/benchmarks.csv")

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

    # Training annotation (no AI yet — we annotate tokens/sec honestly)
    train_rows = df[(df["benchmark"] == "train_singledevice") & (df["metric_name"] == "tokens_per_sec")]
    if not train_rows.empty:
        last = train_rows.iloc[-1]
        tps = float(last["metric_value"])
        params = json.loads(last["params_json"])
        extra = json.loads(last["extra_json"]) if isinstance(last["extra_json"], str) and last["extra_json"] else {}

        label = (
            f"Train step (tiny)\n"
            f"tokens/sec={tps:,.0f}\n"
            f"bs={params.get('batch_size')} seq={params.get('seq_len')}\n"
            f"step_ms={extra.get('step_time_ms', ''):.2f}"
        )

        # Place annotation near ridge line at ~70% of compute roof visually
        x = ridge_ai
        y = peak_compute_gflops * 0.7
        plt.scatter([x], [y], marker="x", s=70, label="Train (annotated)")
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(10, 10))

        print(f"Annotated training: tokens_per_sec={tps:.2f}, step_time_ms={extra.get('step_time_ms')}")

    plt.xlabel("Arithmetic Intensity (FLOP / byte)")
    plt.ylabel("Performance (GFLOP/s)")
    plt.title("CPU Roofline (Ceilings + GEMM + Training Annotation)")
    plt.legend()

    out = Path("plots/roofline_cpu_with_training_annotation.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
