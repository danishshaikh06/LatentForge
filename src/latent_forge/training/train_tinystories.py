
"""Train comparable TinyStories language models using LatentForge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from latent_forge.data.tinystories import load_tinystories_data
from latent_forge.models.decoder import build_transformer_from_config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def choose_precision(requested: str, device: torch.device) -> str:
    if device.type == "cpu":
        if requested not in ("auto", "fp32"):
            raise ValueError("CPU runs require --precision fp32 or auto")
        return "fp32"

    if requested != "auto":
        if requested == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("CUDA device does not support BF16")
        return requested

    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"


def autocast_context(device: torch.device, precision: str):
    if device.type == "cuda" and precision != "fp32":
        dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[precision]
        return torch.autocast("cuda", dtype=dtype)

    return nullcontext()


def make_model_config(args, manifest: dict) -> dict:
    if args.d_model % args.heads:
        raise ValueError("d_model must be divisible by heads")

    if args.attention == "gqa" and args.heads % args.kv_heads:
        raise ValueError("heads must be divisible by kv_heads")

    if args.attention == "mla" and args.mla_rope_dim % 2:
        raise ValueError("MLA rotary dimension must be even")

    # MHA = Hkv == Hq; MQA = Hkv == 1.
    kv_heads = {
        "mha": args.heads,
        "gqa": args.kv_heads,
        "mqa": 1,
        "mla": None,
    }[args.attention]

    head_dim = args.d_model // args.heads

    mla = None
    if args.attention == "mla":
        mla = {
            "q_lora_rank": args.mla_q_rank,
            "kv_lora_rank": args.mla_kv_rank,
            "qk_nope_head_dim": head_dim,
            "qk_rope_head_dim": args.mla_rope_dim,
            "v_head_dim": head_dim,
            "rope_theta": 10000.0,
            "attention_dropout": args.dropout,
            "implementation": "naive",
        }

    return {
        "dataset": "TinyStories",
        "attention_variant": args.attention,
        "attention_type": (
            "mla" if args.attention == "mla" else "gqa"
        ),
        "attention_backend": "sdpa",
        "vocab_size": int(manifest["vocab_size"]),
        "pad_id": int(manifest["pad_id"]),
        "bos_id": int(manifest["bos_id"]),
        "eos_id": int(manifest["eos_id"]),
        "seq_len": args.seq_len,
        "d_model": args.d_model,
        "layers": args.layers,
        "heads": args.heads,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "kv_heads": kv_heads,
        "position_encoding": "rope",
        "rope_theta": 10000.0,
        "rotary_dim": None,
        "norm_type": "rmsnorm",
        "ffn_type": "swiglu",
        "mla": mla,
    }


def token_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pad_id: int,
):
    # Calculate cross-entropy in FP32, even under autocast.
    loss_sum = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=pad_id,
        reduction="sum",
    )
    n_valid = labels.ne(pad_id).sum()
    return loss_sum, n_valid


@torch.inference_mode()
def evaluate(
    model,
    loader,
    device,
    precision,
    pad_id,
    max_batches,
):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        with autocast_context(device, precision):
            # Attention modules apply causal masking internally.
            logits = model(x)
            loss_sum, n_valid = token_loss(logits, y, pad_id)

        total_loss += loss_sum.item()
        total_tokens += int(n_valid.item())

    if total_tokens == 0:
        raise RuntimeError("Evaluation yielded no valid tokens")

    ce = total_loss / total_tokens

    return {
        "ce": ce,
        "ppl": math.exp(ce) if ce < 80 else None,
        "tokens": total_tokens,
    }


def lr_at(
    step: int,
    total: int,
    peak: float,
    minimum: float,
    warmup: int,
) -> float:
    """
    Computes the learning rate using linear warmup followed by
    cosine decay.

    The scheduler consists of two phases:

    1. Linear Warmup:
    Gradually increases the learning rate from a small value
    to the peak learning rate over the specified warmup steps.

    Formula:
        LR(step) = peak * (step + 1) / warmup

    Condition:
        step < warmup

    2. Cosine Decay:
    After warmup, smoothly decreases the learning rate from
    peak to minimum using a cosine annealing schedule.

    First, calculate normalized training progress:

        progress = min(
            1.0,
            (step - warmup) / (total - warmup - 1)
        )

    Then calculate the learning rate:

        LR(step) = minimum + (peak - minimum) * 0.5
                    * (1 + cos(pi * progress))

    Returns:
        float:
            Learning rate corresponding to the current
            training step.

        )
    """
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)

    progress = min(
        1.0,
        (step - warmup) / max(1, total - warmup - 1),
    )
    return minimum + (peak - minimum) * 0.5 * (
        1 + math.cos(math.pi * progress)
    )


def main(args) -> Path:
    if args.steps < 1 or args.batch_size < 1 or args.accum_steps < 1:
        raise ValueError("steps, batch_size and accum_steps must be positive")

    if args.eval_every < 1 or args.val_batches < 1:
        raise ValueError("eval_every and val_batches must be positive")

    if not (0 <= args.min_lr <= args.lr):
        raise ValueError("Expected 0 <= min_lr <= lr")

    if args.warmup_steps < 0 or args.warmup_steps >= args.steps:
        raise ValueError("warmup_steps must be less than steps")

    if args.mode == "overfit" and args.overfit_windows < 1:
        raise ValueError("overfit_windows must be positive")

    if args.mode == "overfit" and args.dropout != 0:
        raise ValueError("Use dropout=0 for the overfit test")

    set_seed(args.seed)

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    precision = choose_precision(args.precision, device)

    # Dataset
    data_root = Path(args.data_dir)

    train_ds, val_ds, tokenizer, manifest = load_tinystories_data(
        data_root,
        seq_len=args.seq_len,
    )

    if sha256_file(data_root / "tokenizer.json") != manifest["tokenizer_sha256"]:
        raise RuntimeError("Tokenizer SHA256 differs from manifest")

    for split in ("train", "validation"):
        expected = manifest.get(split, {}).get("sha256")

        if expected and sha256_file(data_root / f"{split}.bin") != expected:
            raise RuntimeError(f"{split}.bin SHA256 differs from manifest")

    #Model
    cfg = make_model_config(args, manifest)

    model = build_transformer_from_config(
        cfg,
        tgt_vocab_size=cfg["vocab_size"],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())

    if args.mode == "overfit":
        n = min(args.overfit_windows, len(train_ds))

        if n < 1:
            raise ValueError("No training windows")

        # Reuse the exact same windows repeatedly.
        train_ds = Subset(train_ds, list(range(n)))

    if len(train_ds) < args.batch_size:
        raise ValueError("batch_size exceeds available training windows")

    # Isolated generator keeps data order independent of
    # parameter initialization and attention architecture.
    generator = torch.Generator().manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=args.mode == "train",
        generator=generator,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    probe_loader = (
        DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        if args.mode == "overfit"
        else None
    )

    #Run directory
    run_name = (
        args.run_name
        or f"{args.attention}_{args.mode}_seed{args.seed}"
    )

    run_dir = Path(args.output_dir) / run_name

    if (
        (run_dir / "training_config.json").exists()
        and not args.overwrite
    ):
        raise FileExistsError(
            f"{run_dir} exists; use --run-name or --overwrite"
        )

    run_dir.mkdir(parents=True, exist_ok=True)

    cfg.update({
        "dataset_manifest_sha256": sha256_file(
            data_root / "manifest.json"
        ),
        "tokenizer_sha256": manifest["tokenizer_sha256"],
        "data_dir": str(data_root.resolve()),
        "mode": args.mode,
        "seed": args.seed,
        "precision": precision,
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "training_steps": args.steps,
        "trainable_parameters": num_params,
    })

    (run_dir / "training_config.json").write_text(
        json.dumps(cfg, indent=2),
        encoding="utf-8",
    )

    shutil.copy2(
        data_root / "tokenizer.json",
        run_dir / "tokenizer.json",
    )

    shutil.copy2(
        data_root / "manifest.json",
        run_dir / "dataset_manifest.json",
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Loss scaling is useful for FP16, not required for BF16.
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=precision == "fp16",
    )

    iterator = iter(train_loader)

    best_score = float("inf")
    seen_tokens = 0
    train_seconds = 0.0

    metrics_path = run_dir / "metrics.jsonl"

    print(
        f"Device={device} precision={precision} "
        f"parameters={num_params:,}"
    )
    print(
        f"Variant={args.attention} "
        f"train_windows={len(train_ds):,} "
        f"val_windows={len(val_ds):,}"
    )
    print(f"Output={run_dir}")

    # Training loop
    with metrics_path.open("w", encoding="utf-8") as log:

        for step in range(1, args.steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)

            current_lr = lr_at(
                step - 1,
                args.steps,
                args.lr,
                args.min_lr,
                args.warmup_steps,
            )

            for group in optimizer.param_groups:
                group["lr"] = current_lr

            if device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()

            train_ce_sum = 0.0
            step_tokens = 0

            # Gradient accumulation.
            for _ in range(args.accum_steps):
                try:
                    x, y = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    x, y = next(iterator)

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with autocast_context(device, precision):
                    logits = model(x)
                    loss_sum, n_valid = token_loss(
                        logits,
                        y,
                        cfg["pad_id"],
                    )
                    loss = (
                        loss_sum / n_valid.clamp(min=1)
                    ) / args.accum_steps

                if not torch.isfinite(loss).item():
                    raise FloatingPointError(
                        f"Non-finite loss at step {step}"
                    )

                scaler.scale(loss).backward()

                step_tokens += int(n_valid.item())
                train_ce_sum += float(
                    loss_sum.detach().item()
                )

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )

            scaler.step(optimizer)
            scaler.update()

            if device.type == "cuda":
                torch.cuda.synchronize()

            train_seconds += time.perf_counter() - t0
            seen_tokens += step_tokens

            # Validation
            if (
                step == 1
                or step % args.eval_every == 0
                or step == args.steps
            ):
                eval_loader = (
                    probe_loader
                    if args.mode == "overfit"
                    else val_loader
                )

                stats = evaluate(
                    model,
                    eval_loader,
                    device,
                    precision,
                    cfg["pad_id"],
                    None
                    if args.mode == "overfit"
                    else args.val_batches,
                )

                if not math.isfinite(stats["ce"]):
                    raise FloatingPointError(
                        f"Non-finite evaluation loss at step {step}"
                    )

                row = {
                    "step": step,
                    "attention": args.attention,
                    "train_ce_recent": (
                        train_ce_sum / max(step_tokens, 1)
                    ),
                    "eval_split": (
                        "overfit_probe"
                        if args.mode == "overfit"
                        else "validation"
                    ),
                    "eval_ce": stats["ce"],
                    "eval_ppl": stats["ppl"],
                    "eval_tokens": stats["tokens"],
                    "tokens_seen": seen_tokens,
                    "tokens_per_second": (
                        seen_tokens / max(train_seconds, 1e-8)
                    ),
                    "lr": current_lr,
                }

                log.write(
                    json.dumps(row, allow_nan=False) + "\n"
                )
                log.flush()

                print(
                    f"[{args.attention}] "
                    f"step={step}/{args.steps} "
                    f"train_ce={row['train_ce_recent']:.3f} "
                    f"{row['eval_split']}_ce={stats['ce']:.3f} "
                    f"ppl={stats['ppl']}"
                )

                if stats["ce"] < best_score:
                    best_score = stats["ce"]

                    # Plain state_dict for strict model loading.
                    torch.save(
                        model.state_dict(),
                        run_dir / "best_model.pt",
                    )

    #Final checkpoint
    torch.save(
        model.state_dict(),
        run_dir / "last_model.pt",
    )

    summary = {
        "attention": args.attention,
        "mode": args.mode,
        "best_eval_ce": best_score,
        "best_eval_ppl": (
            math.exp(best_score)
            if best_score < 80
            else None
        ),
        "trainable_parameters": num_params,
        "steps": args.steps,
        "tokens_seen": seen_tokens,
        "train_seconds": train_seconds,
        "tokens_per_second": (
            seen_tokens / max(train_seconds, 1e-8)
        ),
        "run_dir": str(run_dir),
    }

    (run_dir / "results.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Finished:", json.dumps(summary, indent=2))

    return run_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train LatentForge on TinyStories"
    )

    parser.add_argument("--data-dir", default="data/tinystories_v1")
    parser.add_argument("--output-dir", default="artifacts/tinystories")
    parser.add_argument(
        "--attention",
        required=True,
        choices=["mha", "gqa", "mqa", "mla"],
    )
    parser.add_argument(
        "--mode",
        choices=["overfit", "train"],
        default="train",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument(
        "--precision",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
    )
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accum-steps", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=64)
    parser.add_argument("--overfit-windows", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--mla-q-rank", type=int, default=96)
    parser.add_argument("--mla-kv-rank", type=int, default=96)
    parser.add_argument("--mla-rope-dim", type=int, default=16)

    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
