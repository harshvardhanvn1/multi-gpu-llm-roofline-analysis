from __future__ import annotations

import argparse
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
    device: str  # "cpu" or "cuda"
    backend: str  # "gloo" or "nccl"
    dtype: str  # "fp32" | "fp16" | "bf16"
    seed: int
    model: str
    use_amp: bool


def _parse_args() -> TrainDDPConfig:
    p = argparse.ArgumentParser(description="DDP training harness (single-node: CPU/GPU).")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--warmup_steps", type=int, default=1)
    p.add_argument("--measure_steps", type=int, default=3)

    cuda_available = torch.cuda.is_available()
    p.add_argument("--device", choices=["cpu", "cuda"], default=("cuda" if cuda_available else "cpu"))
    p.add_argument(
        "--backend",
        choices=["gloo", "nccl"],
        default=("nccl" if cuda_available else "gloo"),
        help="Use nccl for CUDA multi-GPU, gloo for CPU.",
    )
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    p.add_argument("--amp", action="store_true", help="Use autocast on CUDA (recommended for fp16/bf16).")

    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--model", choices=["gpt2_tiny", "gpt2_mediumish"], default="gpt2_tiny")
    a = p.parse_args()

    if a.batch_size <= 0 or a.seq_len <= 0:
        raise ValueError("batch_size and seq_len must be > 0")
    if a.warmup_steps < 0 or a.measure_steps <= 0:
        raise ValueError("warmup_steps must be >= 0 and measure_steps must be > 0")

    return TrainDDPConfig(
        batch_size=int(a.batch_size),
        seq_len=int(a.seq_len),
        warmup_steps=int(a.warmup_steps),
        measure_steps=int(a.measure_steps),
        device=str(a.device),
        backend=str(a.backend),
        dtype=str(a.dtype),
        seed=int(a.seed),
        model=str(a.model),
        use_amp=bool(a.amp),
    )


def _init_distributed(backend: str) -> tuple[int, int]:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend=backend, init_method="env://")
    return dist.get_rank(), dist.get_world_size()


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _make_model(model_name: str, seq_len: int) -> GPT2LMHeadModel:
    # Keep n_positions >= seq_len so attention shapes are correct.
    n_positions = max(seq_len, 256)

    if model_name == "gpt2_tiny":
        cfg = GPT2Config(
            vocab_size=50257,
            n_positions=n_positions,
            n_embd=256,
            n_layer=4,
            n_head=4,
        )
    elif model_name == "gpt2_mediumish":
        # Bigger model so profiling/optimization is meaningful on A100s.
        cfg = GPT2Config(
            vocab_size=50257,
            n_positions=n_positions,
            n_embd=1024,
            n_layer=24,
            n_head=16,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return GPT2LMHeadModel(cfg)


def _synthetic_batch(
    batch_size: int, seq_len: int, vocab_size: int, device: torch.device, seed: int
) -> dict[str, torch.Tensor]:
    # Make the RNG generator live on the same device as the tensor being generated.
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g, device=device)
    return {"input_ids": x, "labels": x}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    cfg = _parse_args()

    # Validate device/backend combo
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("You requested --device cuda, but CUDA is not available.")
    if cfg.backend == "nccl" and cfg.device != "cuda":
        raise ValueError("backend=nccl requires --device cuda")

    rank, world = _init_distributed(cfg.backend)
    lr = _local_rank()

    # Seeding: distinct stream per rank
    torch.manual_seed(cfg.seed + rank)

    # Select device for this process
    if cfg.device == "cuda":
        torch.cuda.set_device(lr)
        device = torch.device("cuda", lr)
    else:
        device = torch.device("cpu")

    # Choose compute dtype and AMP behavior
    if device.type == "cpu":
        compute_dtype = torch.float32
        use_amp = False
    else:
        use_amp = cfg.use_amp
        if cfg.dtype == "fp16":
            compute_dtype = torch.float16
        elif cfg.dtype == "bf16":
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float32

    model = _make_model(cfg.model, cfg.seq_len).to(device)
    model.train()

    # Wrap in DDP
    if device.type == "cuda":
        ddp = DDP(model, device_ids=[lr], output_device=lr)
    else:
        ddp = DDP(model)

    optim = torch.optim.AdamW(ddp.parameters(), lr=5e-4)

    # GradScaler only for CUDA fp16 + AMP (bf16 generally doesn't need it)
    use_grad_scaler = device.type == "cuda" and use_amp and compute_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)

    def train_step(step_seed: int) -> float:
        batch = _synthetic_batch(
            cfg.batch_size,
            cfg.seq_len,
            model.config.vocab_size,
            device,
            step_seed,
        )
        optim.zero_grad(set_to_none=True)

        if device.type == "cuda" and use_amp:
            with torch.autocast(device_type="cuda", dtype=compute_dtype):
                out = ddp(**batch)
                loss = out.loss
            if use_grad_scaler:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()
        else:
            out = ddp(**batch)
            loss = out.loss
            loss.backward()
            optim.step()

        return float(loss.detach().float().cpu().item())

    # Warmup
    for i in range(cfg.warmup_steps):
        _ = train_step(cfg.seed + rank * 10_000 + i)
    _sync(device)

    # Measure
    _sync(device)
    t0 = time.perf_counter()
    last_loss: Optional[float] = None
    for i in range(cfg.measure_steps):
        last_loss = train_step(cfg.seed + rank * 10_000 + 1_000 + i)
    _sync(device)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    step_time_ms_local = (elapsed / cfg.measure_steps) * 1000.0

    # Tokens/sec: compute local, then sum across ranks for global throughput.
    tokens_per_step_local = cfg.batch_size * cfg.seq_len
    tokens_per_sec_local = tokens_per_step_local / (step_time_ms_local / 1000.0)

    tps_tensor = torch.tensor(
        [tokens_per_sec_local], dtype=torch.float64, device=device if device.type == "cuda" else None
    )
    dist.all_reduce(tps_tensor, op=dist.ReduceOp.SUM)
    tokens_per_sec_global = float(tps_tensor.item())

    # Also compute mean step_time across ranks (for reporting consistency)
    step_ms_tensor = torch.tensor(
        [step_time_ms_local], dtype=torch.float64, device=device if device.type == "cuda" else None
    )
    dist.all_reduce(step_ms_tensor, op=dist.ReduceOp.SUM)
    step_time_ms_mean = float(step_ms_tensor.item() / world)

    if rank == 0:
        params = {
            "model": cfg.model,
            "backend": cfg.backend,
            "world_size": world,
            "batch_size": cfg.batch_size,
            "seq_len": cfg.seq_len,
            "warmup_steps": cfg.warmup_steps,
            "measure_steps": cfg.measure_steps,
            "dtype": ("fp32" if device.type == "cpu" else cfg.dtype),
            "device": device.type,
            "amp": bool(device.type == "cuda" and use_amp),
        }
        extra = {
            "step_time_ms_mean": float(step_time_ms_mean),
            "step_time_ms_local_rank0": float(step_time_ms_local),
            "last_loss": float(last_loss) if last_loss is not None else None,
        }

        log_benchmark_result(
            benchmark="train_ddp",
            params=params,
            metric_name="tokens_per_sec_global",
            metric_value=tokens_per_sec_global,
            extra=extra,
        )

        print(
            f"train_ddp ok | device={device.type} backend={cfg.backend} world_size={world} "
            f"step_ms(mean)={step_time_ms_mean:.2f} tokens/sec(global)={tokens_per_sec_global:,.0f} "
            f"last_loss={last_loss:.4f}"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

