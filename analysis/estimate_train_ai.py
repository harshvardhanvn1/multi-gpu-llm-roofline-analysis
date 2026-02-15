from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel


@dataclass(frozen=True)
class Args:
    batch_size: int
    seq_len: int
    n_embd: int
    n_layer: int
    n_head: int
    vocab_size: int
    n_positions: int
    dtype: str
    tokens_per_sec: float
    out: Path


def _parse_args() -> Args:
    p = argparse.ArgumentParser(
        description=(
            "Estimate training arithmetic intensity (FLOPs/byte) and throughput (TFLOP/s) "
            "for the tiny GPT-2 config used in src/train_singledevice.py.\n\n"
            "Notes:\n"
            "- This is an *estimate* meant for roofline plotting, not a perfect profiler.\n"
            "- FLOPs are estimated from transformer math; bytes are a simple lower-bound style model.\n"
        )
    )
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--seq_len", type=int, required=True)

    # Must match your tiny GPT-2
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--vocab_size", type=int, default=50257)
    p.add_argument(
        "--n_positions",
        type=int,
        default=256,
        help="Model max positions. In train_singledevice we use max(seq_len, 256).",
    )

    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    p.add_argument("--tokens_per_sec", type=float, required=True)
    p.add_argument("--out", type=Path, required=True)

    a = p.parse_args()

    if a.batch_size <= 0 or a.seq_len <= 0:
        raise ValueError("batch_size and seq_len must be > 0")
    if a.tokens_per_sec <= 0:
        raise ValueError("tokens_per_sec must be > 0")

    return Args(
        batch_size=a.batch_size,
        seq_len=a.seq_len,
        n_embd=a.n_embd,
        n_layer=a.n_layer,
        n_head=a.n_head,
        vocab_size=a.vocab_size,
        n_positions=max(a.n_positions, a.seq_len),  # mimic train_singledevice's safety
        dtype=a.dtype,
        tokens_per_sec=float(a.tokens_per_sec),
        out=a.out,
    )


def _bytes_per_elem(dtype: str) -> int:
    if dtype == "fp32":
        return 4
    if dtype in ("fp16", "bf16"):
        return 2
    raise ValueError(f"Unknown dtype: {dtype}")


def _build_tiny_gpt2(args: Args) -> GPT2LMHeadModel:
    cfg = GPT2Config(
        vocab_size=args.vocab_size,
        n_positions=args.n_positions,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
    )
    model = GPT2LMHeadModel(cfg)
    model.eval()
    return model


def _count_params(model: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def _estimate_forward_flops_per_step(args: Args) -> float:
    """
    Very standard transformer FLOPs estimate.

    We count multiply-add as 2 FLOPs.
    Per layer forward FLOPs (roughly):
      - Q,K,V projections and output projection: 4 * (2 * B * S * d * d) = 8 * B * S * d^2
      - Attention score matmul (Q @ K^T): 2 * B * S * S * d
      - Attention weighted sum (A @ V):   2 * B * S * S * d
        => total attention matmuls: 4 * B * S^2 * d
      - MLP (d -> 4d -> d): 2 matmuls:
          2 * B * S * d * (4d) + 2 * B * S * (4d) * d = 16 * B * S * d^2

    So per layer forward:
        24 * B * S * d^2 + 4 * B * S^2 * d

    Plus output logits projection (LM head) each token:
        2 * B * S * d * V

    This ignores softmax/loss elementwise FLOPs because the vocab projection dominates anyway.
    """
    B = float(args.batch_size)
    S = float(args.seq_len)
    d = float(args.n_embd)
    V = float(args.vocab_size)
    L = float(args.n_layer)

    per_layer = (24.0 * B * S * (d**2)) + (4.0 * B * (S**2) * d)
    layers_total = L * per_layer

    lm_head = 2.0 * B * S * d * V

    return layers_total + lm_head


def _estimate_train_flops_per_step(args: Args, forward_flops: float) -> float:
    """
    Rule-of-thumb:
      training FLOPs ≈ 3 * forward FLOPs
    because:
      - forward pass
      - backward pass for activations
      - gradient computations for weights

    This is a common approximation for dense networks when you don’t do exact profiling.
    """
    return 3.0 * forward_flops


def _estimate_bytes_per_step_lower_bound(args: Args, param_count: int) -> float:
    """
    A simple, explainable byte model (not perfect, but good for roofline plotting):

    We account for:
      - Reading weights and writing gradients during training
      - A bit of extra traffic for optimizer and intermediate tensors

    Lower-bound-ish heuristic:
      bytes ~= param_bytes * 6

    Why 6?
      - forward reads weights (≈1x)
      - backward reads weights / writes grads (≈2x)
      - optimizer reads/writes states (≈3x)  (AdamW keeps extra buffers)
    Different implementations vary; this gives an order-of-magnitude estimate.

    IMPORTANT:
      - This is intentionally simple and documented in the output JSON.
    """
    bpe = float(_bytes_per_elem(args.dtype))
    param_bytes = float(param_count) * bpe
    return 6.0 * param_bytes


def main() -> None:
    args = _parse_args()

    model = _build_tiny_gpt2(args)
    param_count = _count_params(model)

    fwd_flops_step = _estimate_forward_flops_per_step(args)
    train_flops_step = _estimate_train_flops_per_step(args, fwd_flops_step)

    # Convert tokens/sec -> steps/sec
    tokens_per_step = float(args.batch_size * args.seq_len)
    steps_per_sec = args.tokens_per_sec / tokens_per_step

    flops_per_sec = train_flops_step * steps_per_sec
    tflops = flops_per_sec / 1.0e12

    bytes_per_step = _estimate_bytes_per_step_lower_bound(args, param_count)
    bytes_per_sec = bytes_per_step * steps_per_sec

    # Arithmetic intensity (FLOPs / byte)
    ai = train_flops_step / bytes_per_step

    out_obj = {
        "kind": "train_point_estimate",
        "assumptions": {
            "flops": {
                "mul_add_counts_as": 2,
                "train_flops_multiplier_vs_forward": 3.0,
                "includes_lm_head_vocab_projection": True,
                "ignores_softmax_loss_elementwise_flops": True,
            },
            "bytes": {
                "model": "lower_bound_heuristic",
                "bytes_per_step_formula": "6 * param_bytes(dtype)",
                "notes": "Heuristic to approximate weight+grad+optimizer traffic (AdamW adds extra state).",
            },
        },
        "inputs": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "tokens_per_sec_measured": args.tokens_per_sec,
            "dtype": args.dtype,
            "n_embd": args.n_embd,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "vocab_size": args.vocab_size,
            "n_positions": args.n_positions,
        },
        "model": {
            "param_count": int(param_count),
            "param_bytes": float(param_count) * float(_bytes_per_elem(args.dtype)),
        },
        "derived": {
            "tokens_per_step": tokens_per_step,
            "steps_per_sec": steps_per_sec,
            "forward_flops_per_step": fwd_flops_step,
            "train_flops_per_step": train_flops_step,
            "bytes_per_step_est": bytes_per_step,
            "flops_per_sec_est": flops_per_sec,
            "tflops_est": tflops,
            "bytes_per_sec_est": bytes_per_sec,
            "arithmetic_intensity_flops_per_byte": ai,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_obj, indent=2))
    print(f"Wrote {args.out}")
    print(f"Estimated training point: AI={ai:.3f} FLOPs/byte, Throughput={tflops:.3f} TFLOP/s")


if __name__ == "__main__":
    main()
