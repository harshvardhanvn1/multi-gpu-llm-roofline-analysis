from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch

from src.utils.result_logger import log_benchmark_result


@dataclass(frozen=True)
class GemmConfig:
    n: int
    iters: int
    warmup: int
    dtype: torch.dtype
    device: torch.device
    allow_tf32: bool


def _parse_args() -> GemmConfig:
    p = argparse.ArgumentParser(description="Peak GEMM throughput microbenchmark (CPU/CUDA).")
    p.add_argument("--n", type=int, default=512, help="Matrix dimension N for NxN matmul.")
    p.add_argument("--iters", type=int, default=30, help="Number of timed iterations.")
    p.add_argument("--warmup", type=int, default=5, help="Warmup iterations (not timed).")
    p.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16"], help="Math datatype.")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Execution device.")
    p.add_argument(
        "--allow-tf32",
        action="store_true",
        help="Enable TF32 matmul on CUDA (only relevant for fp32 on Ampere+).",
    )

    args = p.parse_args()

    if args.n <= 0:
        raise ValueError("--n must be > 0")
    if args.iters <= 0:
        raise ValueError("--iters must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    want_cuda = args.device == "cuda"
    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if want_cuda and has_cuda else "cpu")

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16}
    dtype = dtype_map[args.dtype]

    # fp16 on CPU is usually not meaningful for a "peak GEMM" baseline
    if device.type == "cpu" and dtype == torch.float16:
        dtype = torch.float32

    return GemmConfig(
        n=args.n,
        iters=args.iters,
        warmup=args.warmup,
        dtype=dtype,
        device=device,
        allow_tf32=bool(args.allow_tf32),
    )


def _gflops_for_square_gemm(n: int, seconds_per_iter: float) -> float:
    # NxN @ NxN has ~2*N^3 floating point ops (multiply-add counted as 2 ops).
    flops = 2.0 * (n**3)
    return (flops / seconds_per_iter) / 1e9


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_gemm(cfg: GemmConfig) -> float:
    # Configure TF32 if requested (CUDA + fp32 only)
    if cfg.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.allow_tf32 and cfg.dtype == torch.float32)

    a = torch.randn(cfg.n, cfg.n, dtype=cfg.dtype, device=cfg.device)
    b = torch.randn(cfg.n, cfg.n, dtype=cfg.dtype, device=cfg.device)

    # Warmup
    for _ in range(cfg.warmup):
        _ = a @ b
    _sync(cfg.device)

    # Timed
    _sync(cfg.device)
    t0 = time.perf_counter()
    for _ in range(cfg.iters):
        _ = a @ b
    _sync(cfg.device)
    t1 = time.perf_counter()

    avg_sec = (t1 - t0) / cfg.iters
    return _gflops_for_square_gemm(cfg.n, avg_sec)


def main() -> None:
    cfg = _parse_args()
    gflops = run_gemm(cfg)

    dtype_name = "fp32" if cfg.dtype == torch.float32 else "fp16"
    bench_name = f"peak_gemm_{cfg.device.type}"

    params = {
        "n": cfg.n,
        "iters": cfg.iters,
        "warmup": cfg.warmup,
        "dtype": dtype_name,
        "device": cfg.device.type,
        "allow_tf32": cfg.allow_tf32,
    }

    log_benchmark_result(
        benchmark=bench_name,
        params=params,
        metric_name="gflops",
        metric_value=float(gflops),
    )

    print(
        f"{bench_name} n={cfg.n} iters={cfg.iters} warmup={cfg.warmup} "
        f"dtype={dtype_name} device={cfg.device.type} gflops={gflops:.2f}"
    )


if __name__ == "__main__":
    main()
