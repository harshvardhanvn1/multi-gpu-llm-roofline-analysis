from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from src.utils.result_logger import log_benchmark_result


import torch


@dataclass(frozen=True)
class GemmConfig:
    n: int
    iters: int
    warmup: int
    dtype: torch.dtype


def _parse_args() -> GemmConfig:
    p = argparse.ArgumentParser(description="Peak GEMM throughput microbenchmark (CPU for Phase 0).")
    p.add_argument("--n", type=int, default=512, help="Matrix dimension N for NxN matmul.")
    p.add_argument("--iters", type=int, default=30, help="Number of timed iterations.")
    p.add_argument("--warmup", type=int, default=5, help="Warmup iterations (not timed).")
    p.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16"], help="Math datatype.")
    args = p.parse_args()

    if args.n <= 0:
        raise ValueError("--n must be > 0")
    if args.iters <= 0:
        raise ValueError("--iters must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    return GemmConfig(n=args.n, iters=args.iters, warmup=args.warmup, dtype=dtype)


def _gflops_for_square_gemm(n: int, seconds_per_iter: float) -> float:
    # NxN @ NxN has ~2*N^3 floating point ops (multiply-add counted as 2 ops).
    flops = 2.0 * (n ** 3)
    return (flops / seconds_per_iter) / 1e9


def run_gemm(cfg: GemmConfig) -> float:
    a = torch.randn(cfg.n, cfg.n, dtype=cfg.dtype, device="cpu")
    b = torch.randn(cfg.n, cfg.n, dtype=cfg.dtype, device="cpu")

    for _ in range(cfg.warmup):
        _ = a @ b

    t0 = time.perf_counter()
    for _ in range(cfg.iters):
        _ = a @ b
    t1 = time.perf_counter()

    avg_sec = (t1 - t0) / cfg.iters
    return _gflops_for_square_gemm(cfg.n, avg_sec)


def main() -> None:
    cfg = _parse_args()
    gflops = run_gemm(cfg)
    dtype_name = "fp32" if cfg.dtype == torch.float32 else "fp16"

    params = {
        "n": cfg.n,
        "iters": cfg.iters,
        "warmup": cfg.warmup,
        "dtype": dtype_name,
        "device": "cpu",
    }
    log_benchmark_result(
        benchmark="peak_gemm_cpu",
        params=params,
        metric_name="gflops",
        metric_value=float(gflops),
    )
    print(f"peak_gemm_cpu n={cfg.n} iters={cfg.iters} warmup={cfg.warmup} dtype={dtype_name} gflops={gflops:.2f}")



if __name__ == "__main__":
    main()
