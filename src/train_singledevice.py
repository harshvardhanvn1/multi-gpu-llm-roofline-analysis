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


def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Single-device training micro-benchmark (CPU in Phase 0).")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--warmup_steps", type=int, default=2)
    p.add_argument("--measure_steps", type=int, default=5)
    p.add_argument("--dtype", choices=["fp32"], default="fp32", help="CPU baseline uses fp32 for consistency.")
    p.add_argument("--device", choices=["cpu"], default="cpu")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--output", type=Path, default=Path("results/single_cpu_tiny.csv"))
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
    )


def _make_tiny_gpt2(vocab_size: int = 50257, n_positions: int = 256) -> GPT2LMHeadModel:
    # Tiny config for CPU sanity runs (fast, low memory).
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_embd=256,
        n_layer=4,
        n_head=4,
    )
    return GPT2LMHeadModel(cfg)


def _synthetic_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return {"input_ids": input_ids, "labels": input_ids}


def _train_step(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> float:
    out = model(**batch)
    loss = out.loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())


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

    device = torch.device(cfg.device)
    model = _make_tiny_gpt2(n_positions=max(cfg.seq_len, 256)).to(device=device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    # Warmup
    for _ in range(cfg.warmup_steps):
        batch = _synthetic_batch(cfg.batch_size, cfg.seq_len, model.config.vocab_size, device)
        _ = _train_step(model, batch, optimizer)

    # Measure
    start = time.perf_counter()
    last_loss: Optional[float] = None
    for _ in range(cfg.measure_steps):
        batch = _synthetic_batch(cfg.batch_size, cfg.seq_len, model.config.vocab_size, device)
        last_loss = _train_step(model, batch, optimizer)
    end = time.perf_counter()

    elapsed = end - start
    step_time_ms = (elapsed / cfg.measure_steps) * 1000.0
    tokens_per_step = cfg.batch_size * cfg.seq_len
    tokens_per_sec = tokens_per_step / (step_time_ms / 1000.0)

    params = {
        "device": cfg.device,
        "dtype": cfg.dtype,
        "batch_size": cfg.batch_size,
        "seq_len": cfg.seq_len,
        "warmup_steps": cfg.warmup_steps,
        "measure_steps": cfg.measure_steps,
        "model": "gpt2_tiny",
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
        "batch_size",
        "seq_len",
        "warmup_steps",
        "measure_steps",
        "step_time_ms",
        "tokens_per_sec",
        "last_loss",
    ]
    row = {
        "device": cfg.device,
        "dtype": cfg.dtype,
        "batch_size": cfg.batch_size,
        "seq_len": cfg.seq_len,
        "warmup_steps": cfg.warmup_steps,
        "measure_steps": cfg.measure_steps,
        "step_time_ms": step_time_ms,
        "tokens_per_sec": tokens_per_sec,
        "last_loss": last_loss if last_loss is not None else "",
    }
    _write_csv_row(cfg.output, header, row)

    print("Single-device Training Benchmark (CPU tiny)")
    print(f"Device: {cfg.device} | dtype: {cfg.dtype}")
    print(f"Batch size: {cfg.batch_size} | Seq len: {cfg.seq_len}")
    print(f"Step time: {step_time_ms:.2f} ms | Tokens/sec: {tokens_per_sec:,.0f}")
    print(f"Wrote {cfg.output}")


if __name__ == "__main__":
    main()
