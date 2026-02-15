from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import GPT2Config, GPT2LMHeadModel


@dataclass(frozen=True)
class ProfileDDPConfig:
    batch_size: int
    seq_len: int
    warmup_steps: int
    profile_steps: int
    device: str
    backend: str
    dtype: str
    seed: int
    model: str
    use_amp: bool
    out_dir: str


def _parse_args() -> ProfileDDPConfig:
    p = argparse.ArgumentParser(description="DDP profiler harness (single-node). Writes a Chrome trace to results/.")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--warmup_steps", type=int, default=1)
    p.add_argument("--profile_steps", type=int, default=5)

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
    p.add_argument("--out_dir", type=str, default="results")
    a = p.parse_args()

    if a.batch_size <= 0 or a.seq_len <= 0:
        raise ValueError("batch_size and seq_len must be > 0")
    if a.warmup_steps < 0 or a.profile_steps <= 0:
        raise ValueError("warmup_steps must be >= 0 and profile_steps must be > 0")

    return ProfileDDPConfig(
        batch_size=int(a.batch_size),
        seq_len=int(a.seq_len),
        warmup_steps=int(a.warmup_steps),
        profile_steps=int(a.profile_steps),
        device=str(a.device),
        backend=str(a.backend),
        dtype=str(a.dtype),
        seed=int(a.seed),
        model=str(a.model),
        use_amp=bool(a.amp),
        out_dir=str(a.out_dir),
    )


def _init_distributed(backend: str) -> tuple[int, int]:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend=backend, init_method="env://")
    return dist.get_rank(), dist.get_world_size()


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _make_model(model_name: str, seq_len: int) -> GPT2LMHeadModel:
    if model_name == "gpt2_tiny":
        cfg = GPT2Config(
            vocab_size=50257,
            n_positions=max(seq_len, 256),
            n_embd=256,
            n_layer=4,
            n_head=4,
        )
        return GPT2LMHeadModel(cfg)

    if model_name == "gpt2_mediumish":
        # Medium-ish GPT-2 style model to better stress A100s/NVLink.
        cfg = GPT2Config(
            vocab_size=50257,
            n_positions=max(seq_len, 1024),
            n_embd=1024,
            n_layer=24,
            n_head=16,
        )
        return GPT2LMHeadModel(cfg)

    raise ValueError(f"Unsupported model: {model_name}")


def _synthetic_batch(
    batch_size: int, seq_len: int, vocab_size: int, device: torch.device, seed: int
) -> dict[str, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g, device=device)
    return {"input_ids": x, "labels": x}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    cfg = _parse_args()

    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("You requested --device cuda, but CUDA is not available.")
    if cfg.backend == "nccl" and cfg.device != "cuda":
        raise ValueError("backend=nccl requires --device cuda")

    rank, world = _init_distributed(cfg.backend)
    lr = _local_rank()

    torch.manual_seed(cfg.seed + rank)

    if cfg.device == "cuda":
        torch.cuda.set_device(lr)
        device = torch.device("cuda", lr)
    else:
        device = torch.device("cpu")

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

    if device.type == "cuda":
        ddp = DDP(model, device_ids=[lr], output_device=lr)
    else:
        ddp = DDP(model)

    optim = torch.optim.AdamW(ddp.parameters(), lr=5e-4)

    # GradScaler only for CUDA fp16 + AMP (bf16 generally doesn't need it)
    use_grad_scaler = device.type == "cuda" and use_amp and compute_dtype == torch.float16
    if use_grad_scaler:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    else:
        scaler = torch.amp.GradScaler("cuda", enabled=False)

    def train_step(step_seed: int) -> float:
        batch = _synthetic_batch(cfg.batch_size, cfg.seq_len, model.config.vocab_size, device, step_seed)
        optim.zero_grad(set_to_none=True)

        if device.type == "cuda" and use_amp:
            with torch.autocast(device_type="cuda", dtype=compute_dtype):
                with record_function("forward"):
                    out = ddp(**batch)
                    loss = out.loss
            with record_function("backward"):
                if use_grad_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            with record_function("optimizer_step"):
                if use_grad_scaler:
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
        else:
            with record_function("forward"):
                out = ddp(**batch)
                loss = out.loss
            with record_function("backward"):
                loss.backward()
            with record_function("optimizer_step"):
                optim.step()

        return float(loss.detach().float().cpu().item())

    # Warmup (not profiled)
    for i in range(cfg.warmup_steps):
        _ = train_step(cfg.seed + rank * 10_000 + i)
    _sync(device)

    # Profile
    os.makedirs(cfg.out_dir, exist_ok=True)
    trace_path = os.path.join(cfg.out_dir, f"trace_ddp_rank{rank}.json")

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    _sync(device)
    t0 = time.perf_counter()
    last_loss: Optional[float] = None

    with profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for i in range(cfg.profile_steps):
            last_loss = train_step(cfg.seed + rank * 10_000 + 1_000 + i)
            _sync(device)
            prof.step()

    t1 = time.perf_counter()

    # Write trace for each rank (Chrome trace format)
    prof.export_chrome_trace(trace_path)

    if rank == 0:
        elapsed = t1 - t0
        step_ms = (elapsed / cfg.profile_steps) * 1000.0
        print(
            f"profile_ddp ok | device={device.type} backend={cfg.backend} world_size={world} "
            f"profile_steps={cfg.profile_steps} step_ms~{step_ms:.2f} "
            f"trace_dir={cfg.out_dir} (one trace per rank) last_loss={last_loss:.4f}"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
