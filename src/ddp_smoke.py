from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Config, GPT2LMHeadModel


@dataclass(frozen=True)
class Config:
    batch_size: int
    seq_len: int
    steps: int
    device: str  # "cpu" or "cuda"
    backend: str  # "gloo" or "nccl"
    seed: int


def _parse_args() -> Config:
    p = argparse.ArgumentParser(description="DDP smoke test (single node, multi-process).")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--steps", type=int, default=2)

    # Default behavior: if CUDA is available, use CUDA+NCCL; otherwise CPU+Gloo.
    cuda_available = torch.cuda.is_available()
    p.add_argument("--device", choices=["cpu", "cuda"], default=("cuda" if cuda_available else "cpu"))
    p.add_argument("--backend", choices=["gloo", "nccl"], default=("nccl" if cuda_available else "gloo"))

    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    return Config(
        batch_size=int(args.batch_size),
        seq_len=int(args.seq_len),
        steps=int(args.steps),
        device=str(args.device),
        backend=str(args.backend),
        seed=int(args.seed),
    )


def _make_tiny_gpt2(n_positions: int) -> GPT2LMHeadModel:
    cfg = GPT2Config(
        n_embd=256,
        n_layer=4,
        n_head=4,
        n_positions=max(n_positions, 256),
    )
    return GPT2LMHeadModel(cfg)


def _synthetic_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return {"input_ids": x, "labels": x}


def main() -> None:
    cfg = _parse_args()
    torch.manual_seed(cfg.seed)

    # env:// init requires these. torchrun usually provides them; defaults help local runs.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    if cfg.backend == "nccl" and cfg.device != "cuda":
        raise ValueError("backend=nccl requires --device cuda")
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("You requested --device cuda, but CUDA is not available.")

    dist.init_process_group(backend=cfg.backend, init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if cfg.device == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    model = _make_tiny_gpt2(cfg.seq_len).to(device)
    model.train()

    # For single-node torchrun, it's safe to pin each process to its GPU via device_ids.
    if cfg.device == "cuda":
        ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    else:
        ddp_model = DDP(model)

    optim = torch.optim.AdamW(ddp_model.parameters(), lr=5e-4)

    # Warmup one step to reduce first-iteration overhead noise (especially on CUDA).
    batch = _synthetic_batch(cfg.batch_size, cfg.seq_len, model.config.vocab_size, device)
    out = ddp_model(**batch)
    loss = out.loss
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()
    if cfg.device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    last_loss = None
    for _ in range(cfg.steps):
        batch = _synthetic_batch(cfg.batch_size, cfg.seq_len, model.config.vocab_size, device)
        out = ddp_model(**batch)
        loss = out.loss
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        last_loss = float(loss.detach().float().cpu().item())
    if cfg.device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    if rank == 0:
        print(
            f"ddp_smoke ok | device={cfg.device} backend={cfg.backend} "
            f"world_size={world_size} steps={cfg.steps} last_loss={last_loss:.4f} "
            f"elapsed_ms={(t1 - t0) * 1000:.2f}"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
