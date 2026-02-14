from __future__ import annotations

import subprocess
import sys


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    # Collect a few samples to reduce noise (ceilings use max).
    for _ in range(5):
        _run([sys.executable, "-m", "benchmarks.peak_gemm", "--n", "512", "--iters", "30", "--warmup", "5", "--dtype", "fp32"])
    for _ in range(5):
        _run([sys.executable, "-m", "benchmarks.peak_bw", "--size-mb", "512", "--iters", "50", "--warmup", "10", "--dtype", "fp32"])

    _run([sys.executable, "analysis/aggregate_ceilings.py"])
    _run([sys.executable, "analysis/plot_roofline.py"])
    _run([sys.executable, "analysis/plot_roofline_with_points.py"])


if __name__ == "__main__":
    main()
