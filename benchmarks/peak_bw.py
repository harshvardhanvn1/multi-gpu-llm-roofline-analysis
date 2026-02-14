from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from src.utils.result_logger import log_benchmark_result

import torch


@dataclass(frozen=True)
class BwConfig:
    size_mb: int
    iters: int
    warmup: int
    dtype: torch.dtype


def _parse_args() -> BwConfig:
    p = argparse.ArgumentParser(description="Peak memory bandwidth microbenchmark (CPU for Phase 0).")
    p.add_argument("--size-mb", type=int, default=512, help="Tensor size in MB (should exceed LLC cache).")
    p.add_argument("--iters", type=int, default=50, help="Number of timed iterations.")
    p.add_argument("--warmup", type=int, default=10, help="Warmup iterations (not timed).")
    p.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16"], help="Tensor dtype.")
    args = p.parse_args()

    if args.size_mb <= 0:
        raise ValueError("--size-mb must be > 0")
    if args.iters <= 0:
        raise ValueError("--iters must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    return BwConfig(size_mb=args.size_mb, iters=args.iters, warmup=args.warmup, dtype=dtype)


def _bytes_per_element(dtype: torch.dtype) -> int:
    if dtype == torch.float32:
        return 4
    if dtype == torch.float16:
        return 2
    raise ValueError(f"Unsupported dtype: {dtype}")


def run_copy_bandwidth(cfg: BwConfig) -> float:
    bytes_total = cfg.size_mb * 1024 * 1024
    elem_bytes = _bytes_per_element(cfg.dtype)
    numel = bytes_total // elem_bytes

    src = torch.randn(numel, dtype=cfg.dtype, device="cpu")
    dst = torch.empty_like(src)

    # Warmup (helps avoid one-time initialization effects)
    for _ in range(cfg.warmup):
        dst.copy_(src)

    # Timed copies
    t0 = time.perf_counter()
    for _ in range(cfg.iters):
        dst.copy_(src)
    t1 = time.perf_counter()

    avg_sec = (t1 - t0) / cfg.iters

    # Copy touches memory for both read (src) and write (dst) => ~2x bytes moved.
    bytes_moved = 2.0 * bytes_total
    gb_per_s = (bytes_moved / avg_sec) / 1e9
    return gb_per_s


def main() -> None:
    cfg = _parse_args()
    gbps = run_copy_bandwidth(cfg)
    dtype_name = "fp32" if cfg.dtype == torch.float32 else "fp16"

    params = {
        "op": "copy",
        "size_mb": cfg.size_mb,
        "iters": cfg.iters,
        "warmup": cfg.warmup,
        "dtype": dtype_name,
        "device": "cpu",
    }
    log_benchmark_result(
        benchmark="peak_bw_cpu",
        params=params,
        metric_name="gbps",
        metric_value=float(gbps),
    )

    print(
        f"peak_bw_cpu op=copy size_mb={cfg.size_mb} iters={cfg.iters} warmup={cfg.warmup} "
        f"dtype={dtype_name} gbps={gbps:.2f}"
    )


if __name__ == "__main__":
    main()
