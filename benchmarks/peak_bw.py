from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch

from src.utils.result_logger import log_benchmark_result


@dataclass(frozen=True)
class BwConfig:
    size_mb: int
    iters: int
    warmup: int
    dtype: torch.dtype
    device: torch.device


def _parse_args() -> BwConfig:
    p = argparse.ArgumentParser(description="Peak memory bandwidth microbenchmark (CPU/CUDA).")
    p.add_argument("--size-mb", type=int, default=512, help="Tensor size in MB.")
    p.add_argument("--iters", type=int, default=50, help="Number of timed iterations.")
    p.add_argument("--warmup", type=int, default=10, help="Warmup iterations (not timed).")
    p.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16"], help="Tensor dtype.")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Execution device.")
    args = p.parse_args()

    if args.size_mb <= 0:
        raise ValueError("--size-mb must be > 0")
    if args.iters <= 0:
        raise ValueError("--iters must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    want_cuda = args.device == "cuda"
    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if want_cuda and has_cuda else "cpu")

    dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    if device.type == "cpu" and dtype == torch.float16:
        # fp16 bandwidth on CPU is not a stable/meaningful "peak" baseline
        dtype = torch.float32

    return BwConfig(size_mb=args.size_mb, iters=args.iters, warmup=args.warmup, dtype=dtype, device=device)


def _bytes_per_element(dtype: torch.dtype) -> int:
    if dtype == torch.float32:
        return 4
    if dtype == torch.float16:
        return 2
    raise ValueError(f"Unsupported dtype: {dtype}")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_copy_bandwidth(cfg: BwConfig) -> float:
    bytes_total = cfg.size_mb * 1024 * 1024
    elem_bytes = _bytes_per_element(cfg.dtype)
    numel = bytes_total // elem_bytes

    src = torch.randn(numel, dtype=cfg.dtype, device=cfg.device)
    dst = torch.empty_like(src)

    # Warmup
    for _ in range(cfg.warmup):
        dst.copy_(src)
    _sync(cfg.device)

    # Timed copies
    _sync(cfg.device)
    t0 = time.perf_counter()
    for _ in range(cfg.iters):
        dst.copy_(src)
    _sync(cfg.device)
    t1 = time.perf_counter()

    avg_sec = (t1 - t0) / cfg.iters

    # copy = read src + write dst => ~2x bytes moved
    bytes_moved = 2.0 * bytes_total
    gb_per_s = (bytes_moved / avg_sec) / 1e9
    return gb_per_s


def main() -> None:
    cfg = _parse_args()
    gbps = run_copy_bandwidth(cfg)
    dtype_name = "fp32" if cfg.dtype == torch.float32 else "fp16"

    bench_name = f"peak_bw_{cfg.device.type}"
    params = {
        "op": "copy",
        "size_mb": cfg.size_mb,
        "iters": cfg.iters,
        "warmup": cfg.warmup,
        "dtype": dtype_name,
        "device": cfg.device.type,
    }

    log_benchmark_result(
        benchmark=bench_name,
        params=params,
        metric_name="gbps",
        metric_value=float(gbps),
    )

    print(
        f"{bench_name} op=copy size_mb={cfg.size_mb} iters={cfg.iters} warmup={cfg.warmup} "
        f"dtype={dtype_name} device={cfg.device.type} gbps={gbps:.2f}"
    )


if __name__ == "__main__":
    main()
