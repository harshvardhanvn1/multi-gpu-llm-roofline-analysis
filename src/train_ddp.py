from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Config, GPT2LMHeadModel

from src.utils.result_logger import log_benchmark_result


@dataclass(frozen=True)
class TrainDDPConfig:
    batch_size: int
    seq_len: int
    warmup_steps: int
    measure_steps: int
    backend: str
    dtype: str
    seed: int
    model: str


def _parse_args() -> TrainDDPConfig:
    p = argparse.ArgumentParser(description="DDP training harness (CPU=gloo now; GPU=nccl later).")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--warmup_steps", type=int, default=1)
    p.add_argument("--measure_steps", type=int, default=3)
    p.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    p.add_argument("--dtype", choices=["fp32"], default="fp32")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--model", choices=["gpt2_tiny"], default="gpt2_tiny")
    a = p.parse_args()

    if a.batch_size <= 0 or a.seq_len <= 0:
        raise ValueError("batch_size and seq_len must be > 0")
    if a.warmup_steps < 0 or a.measure_steps <= 0:
        raise ValueError("warmup_steps must be >= 0 and measure_steps must be > 0")

    return TrainDDPConfig(
        batch_size=a.batch_size,
        seq_len=a.seq_len,
        warmup_steps=a.warmup_steps,
        measure_steps=a.measure_steps,
        backend=a.backend,
        dtype=a.dtype,
        seed=a.seed,
        model=a.model,
    )


def _get_rank_world() -> tuple[int, int]:
    # torchrun sets these; for local CPU single-process we default safely.
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world


def _init_distributed(backend: str) -> tuple[int, int]:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    # init_method env:// uses MASTER_ADDR/PORT + RANK/WORLD_SIZE
    dist.init_process_group(backend=backend, init_method="env://")
    return dist.get_rank(), dist.get_world_size()


def _make_model(model_name: str, seq_len: int) -> GPT2LMHeadModel:
    if model_name != "gpt2_tiny":
        raise ValueError(f"Unsupported model: {model_name}")

    cfg = GPT2Config(
        vocab_size=50257,
        n_positions=max(seq_len, 256),
        n_embd=256,
        n_layer=4,
        n_head=4,
    )
    return GPT2LMHeadModel(cfg)


def _synthetic_batch(
    batch_size: int, seq_len: int, vocab_size: int, device: torch.device, seed: int
) -> dict[str, torch.Tensor]:
    # Rank-specific seed -> different synthetic data on each rank (important when world_size>1).
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g, device=device)
    return {"input_ids": x, "labels": x}


def _train_step(model: torch.nn.Module, batch: dict[str, torch.Tensor], optim: torch.optim.Optimizer) -> float:
    out = model(**batch)
    loss = out.loss
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()
    return float(loss.detach().cpu().item())


def main() -> None:
    cfg = _parse_args()

    # For correctness when scaling: each rank gets a distinct seed stream.
    base_seed = cfg.seed
    rank_env, world_env = _get_rank_world()
    torch.manual_seed(base_seed + rank_env)

    rank, world = _init_distributed(cfg.backend)

    device = torch.device("cpu")  # CPU for Phase 0
    model = _make_model(cfg.model, cfg.seq_len).to(device)
    model.train()

    ddp = DDP(model)
    optim = torch.optim.AdamW(ddp.parameters(), lr=5e-4)

    # Warmup
    for i in range(cfg.warmup_steps):
        batch = _synthetic_batch(cfg.batch_size, cfg.seq_len, model.config.vocab_size, device, base_seed + rank * 10_000 + i)
        _ = _train_step(ddp, batch, optim)

    # Measure
    t0 = time.perf_counter()
    last_loss: Optional[float] = None
    for i in range(cfg.measure_steps):
        batch = _synthetic_batch(cfg.batch_size, cfg.seq_len, model.config.vocab_size, device, base_seed + rank * 10_000 + 1_000 + i)
        last_loss = _train_step(ddp, batch, optim)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    step_time_ms = (elapsed / cfg.measure_steps) * 1000.0

    # Global tokens/sec (sum across ranks). With world_size=1, same as local.
    tokens_per_step_local = cfg.batch_size * cfg.seq_len
    tokens_per_sec_local = tokens_per_step_local / (step_time_ms / 1000.0)

    tps_tensor = torch.tensor([tokens_per_sec_local], dtype=torch.float64)
    dist.all_reduce(tps_tensor, op=dist.ReduceOp.SUM)  # sum across ranks
    tokens_per_sec_global = float(tps_tensor.item())

    if rank == 0:
        params = {
            "model": cfg.model,
            "backend": cfg.backend,
            "world_size": world,
            "batch_size": cfg.batch_size,
            "seq_len": cfg.seq_len,
            "warmup_steps": cfg.warmup_steps,
            "measure_steps": cfg.measure_steps,
            "dtype": cfg.dtype,
            "device": "cpu",
        }
        extra = {"step_time_ms": float(step_time_ms), "last_loss": float(last_loss) if last_loss is not None else None}

        log_benchmark_result(
            benchmark="train_ddp",
            params=params,
            metric_name="tokens_per_sec_global",
            metric_value=tokens_per_sec_global,
            extra=extra,
        )

        print(
            f"train_ddp ok | backend={cfg.backend} world_size={world} "
            f"step_ms={step_time_ms:.2f} tokens/sec(global)={tokens_per_sec_global:,.0f} last_loss={last_loss:.4f}"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
