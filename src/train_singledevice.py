from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from src.utils.result_logger import log_benchmark_result


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    seq_len: int
    warmup_steps: int
    measure_steps: int
    dtype: str
    output: Path
    device: str
    seed: int
    use_amp: bool
    model: str


def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Single-device training micro-benchmark (CPU/CUDA).")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--warmup_steps", type=int, default=2)
    p.add_argument("--measure_steps", type=int, default=5)
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--output", type=Path, default=Path("results/single.csv"))
    p.add_argument("--amp", action="store_true", help="Use autocast on CUDA (recommended for fp16/bf16).")
    p.add_argument("--model", choices=["gpt2_tiny", "gpt2_mediumish"], default="gpt2_tiny")
    args = p.parse_args()

    if args.batch_size <= 0 or args.seq_len <= 0:
        raise ValueError("batch_size and seq_len must be > 0")
    if args.warmup_steps < 0 or args.measure_steps <= 0:
        raise ValueError("warmup_steps must be >= 0 and measure_steps must be > 0")

    return TrainConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        dtype=args.dtype,
        output=args.output,
        device=args.device,
        seed=args.seed,
        use_amp=bool(args.amp),
        model=str(args.model),
    )


def _make_model(model_name: str, seq_len: int, vocab_size: int = 50257) -> GPT2LMHeadModel:
    # Keep n_positions >= seq_len so attention shapes are correct.
    n_positions = max(seq_len, 256)

    if model_name == "gpt2_tiny":
        cfg = GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=256,
            n_layer=4,
            n_head=4,
        )
    elif model_name == "gpt2_mediumish":
        # Bigger but still safe on 2×80GB for profiling/benchmarks.
        # This increases compute + gradient sizes so comm/overhead becomes measurable.
        cfg = GPT2Config(
            vocab_size=vocab_size,
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
    g = torch.Generator(device=device).manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g, device=device)
    return {"input_ids": input_ids, "labels": input_ids}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _write_csv_row(path: Path, header: list[str], row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    cfg = _parse_args()
    torch.manual_seed(cfg.seed)

    want_cuda = cfg.device == "cuda"
    if want_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "You requested --device cuda, but torch.cuda.is_available() is False. "
            "Fix the CUDA/PyTorch setup or run with --device cpu."
        )
    device = torch.device("cuda" if want_cuda else "cpu")

    # Choose compute dtype
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

    model = _make_model(cfg.model, cfg.seq_len).to(device=device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    use_grad_scaler = device.type == "cuda" and compute_dtype == torch.float16 and use_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)

    def train_step(step_seed: int) -> float:
        batch = _synthetic_batch(cfg.batch_size, cfg.seq_len, model.config.vocab_size, device, step_seed)
        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda" and use_amp:
            with torch.autocast(device_type="cuda", dtype=compute_dtype):
                out = model(**batch)
                loss = out.loss
            if use_grad_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optimizer.step()

        return float(loss.detach().float().cpu().item())

    # Warmup
    for i in range(cfg.warmup_steps):
        _ = train_step(cfg.seed + i)
    _sync(device)

    # Measure
    _sync(device)
    start = time.perf_counter()
    last_loss: Optional[float] = None
    for i in range(cfg.measure_steps):
        last_loss = train_step(cfg.seed + 1_000 + i)
    _sync(device)
    end = time.perf_counter()

    elapsed = end - start
    step_time_ms = (elapsed / cfg.measure_steps) * 1000.0
    tokens_per_step = cfg.batch_size * cfg.seq_len
    tokens_per_sec = tokens_per_step / (step_time_ms / 1000.0)

    dtype_name = "fp32" if device.type == "cpu" else cfg.dtype

    params = {
        "device": device.type,
        "dtype": dtype_name,
        "batch_size": cfg.batch_size,
        "seq_len": cfg.seq_len,
        "warmup_steps": cfg.warmup_steps,
        "measure_steps": cfg.measure_steps,
        "model": cfg.model,
        "amp": bool(device.type == "cuda" and use_amp),
    }

    log_benchmark_result(
        benchmark="train_singledevice",
        params=params,
        metric_name="tokens_per_sec",
        metric_value=float(tokens_per_sec),
        extra={"step_time_ms": float(step_time_ms), "last_loss": float(last_loss) if last_loss is not None else None},
    )

    header = [
        "device",
        "dtype",
        "model",
        "batch_size",
        "seq_len",
        "warmup_steps",
        "measure_steps",
        "step_time_ms",
        "tokens_per_sec",
        "last_loss",
    ]
    row = {
        "device": device.type,
        "dtype": dtype_name,
        "model": cfg.model,
        "batch_size": cfg.batch_size,
        "seq_len": cfg.seq_len,
        "warmup_steps": cfg.warmup_steps,
        "measure_steps": cfg.measure_steps,
        "step_time_ms": step_time_ms,
        "tokens_per_sec": tokens_per_sec,
        "last_loss": last_loss if last_loss is not None else "",
    }
    _write_csv_row(cfg.output, header, row)

    print(f"Single-device Training Benchmark ({cfg.model})")
    print(f"Device: {device.type} | dtype: {dtype_name} | amp: {params['amp']}")
    print(f"Batch size: {cfg.batch_size} | Seq len: {cfg.seq_len}")
    print(f"Step time: {step_time_ms:.2f} ms | Tokens/sec: {tokens_per_sec:,.0f}")
    print(f"Wrote {cfg.output}")


if __name__ == "__main__":
    main()
