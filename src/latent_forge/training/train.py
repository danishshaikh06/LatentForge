import argparse
import json
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from latent_forge.core.paths import (
    CONFIGS_DIR,
    BEST_MODEL_FILE,
    TOKENS_WITH_SPECIAL_FILE,
    TRAINING_CONFIG_FILE,
    ensure_artifacts_dir,
)
from latent_forge.models.decoder import build_transformer
from latent_forge.training.inference import generate_completion, load_tokenizer
from latent_forge.models.decoder import (
    build_transformer_from_config,
)

with (CONFIGS_DIR / "training.json").open(encoding="utf-8") as config_file:
    TRAINING_PROFILES = json.load(config_file)

TRAINING_CONFIG = TRAINING_PROFILES["cuda" if torch.cuda.is_available() else "cpu"]
SPECIAL_TOKENS = TRAINING_PROFILES["special_tokens"]


SAMPLE_PROMPTS = [
    "Give me a hello world program in C++ with explanation.",
    "Give me a black hole simulation code snippet in C++.",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"


class TokenDataset(Dataset):
    def __init__(self, tokens, seq_len, stride):
        self.tokens = tokens
        self.seq_len = seq_len
        self.stride = stride

    def __len__(self):
        return max(((len(self.tokens) - self.seq_len - 1) // self.stride) + 1, 0)

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.tokens[start : start + self.seq_len]
        y = self.tokens[start + 1 : start + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the decoder-only C++ code generator.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a short verification training pass on a small subset of tokens.",
    )
    parser.add_argument(
        "--max-train-tokens",
        type=int,
        default=None,
        help="Limit the number of tokens used before the train/val split.",
    )
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=None,
        help="Limit optimizer steps per epoch for quick checks.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the default epoch count.",
    )
    parser.add_argument(
        "--attention-type",
        choices=("mha", "gqa", "mla"),
        default=None,
        help="Override the configured attention architecture",
    )
    return parser.parse_args()


def build_runtime_config(args):
    runtime_config = dict(TRAINING_CONFIG)

    if args.attention_type is not None:
        runtime_config["attention_type"] = args.attention_type

        if args.attention_type == "mla":
            runtime_config["kv_heads"] = None
            runtime_config["position_encoding"] = "rope"

            if "mla" not in runtime_config:
                raise ValueError(
                    "MLA configuration is missing"
                )

    elif args.attention_type == "mha":
        runtime_config["kv_heads"] = None
        runtime_config["position_encoding"] = "sinusoidal"

    elif args.attention_type == "gqa":
        if runtime_config.get("kv_heads") is None:
            raise ValueError(
                "GQA requires kv_heads in training.json"
            )

    if args.epochs is not None:
        runtime_config["epochs"] = args.epochs

    if args.smoke_test:
        runtime_config["epochs"] = 1 if args.epochs is None else args.epochs
        runtime_config["sample_every_epochs"] = 1
        runtime_config["max_train_tokens"] = (
            args.max_train_tokens if args.max_train_tokens is not None else 200_000
        )
        runtime_config["max_steps_per_epoch"] = (
            args.max_steps_per_epoch if args.max_steps_per_epoch is not None else 100
        )
    else:
        runtime_config["sample_every_epochs"] = 5
        runtime_config["max_train_tokens"] = args.max_train_tokens
        runtime_config["max_steps_per_epoch"] = args.max_steps_per_epoch

    return runtime_config


def get_lr(step, base_lr, warmup_steps=1000):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    return base_lr * (0.95 ** (step / 10000))


def maybe_trim_tokens(tokens, max_train_tokens, bos_id, eos_id):
    if max_train_tokens is None or max_train_tokens >= len(tokens):
        return tokens

    core_tokens = tokens
    has_bos = len(tokens) > 0 and int(tokens[0]) == bos_id
    has_eos = len(tokens) > 0 and int(tokens[-1]) == eos_id

    if has_bos:
        core_tokens = core_tokens[1:]
    if has_eos:
        core_tokens = core_tokens[:-1]

    usable_core_len = max(max_train_tokens - int(has_bos) - int(has_eos), 1)
    trimmed_core = core_tokens[:usable_core_len]

    rebuilt = trimmed_core
    if has_bos:
        rebuilt = np.concatenate(([bos_id], rebuilt))
    if has_eos:
        rebuilt = np.concatenate((rebuilt, [eos_id]))

    return rebuilt


def run_sample_generations(model, tokenizer, seq_len, pad_id, eos_id, bos_id, device):
    print("\nSample generations:")
    for prompt in SAMPLE_PROMPTS:
        sample = generate_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            seq_len=seq_len,
            pad_id=pad_id,
            eos_id=eos_id,
            bos_id=bos_id,
            device=device,
            max_new_tokens=120,
            temperature=0.8,
            top_k=40,
        )
        print(f"\nPrompt: {prompt}\nResponse: {sample}\n")


def main():
    args = parse_args()
    runtime_config = build_runtime_config(args)

    seq_len = runtime_config["seq_len"]
    stride = runtime_config["stride"]
    batch_size = runtime_config["batch_size"]
    epochs = runtime_config["epochs"]
    lr = runtime_config["lr"]
    accum_steps = runtime_config["accum_steps"]
    sample_every_epochs = runtime_config["sample_every_epochs"]

    artifacts_root = ensure_artifacts_dir()

    pad_id = SPECIAL_TOKENS["pad_id"]
    bos_id = SPECIAL_TOKENS["bos_id"]
    eos_id = SPECIAL_TOKENS["eos_id"]

    tokens = np.load(TOKENS_WITH_SPECIAL_FILE)
    print("Total tokens:", len(tokens))

    tokens = maybe_trim_tokens(tokens, runtime_config["max_train_tokens"], bos_id, eos_id)
    print("Tokens used for this run:", len(tokens))

    max_token_id = int(tokens.max())
    vocab_size = max(max_token_id + 1, eos_id + 1)

    assert pad_id < vocab_size, "PAD_ID must be inside vocab range!"

    attention_type = runtime_config.get("attention_type")

    if attention_type is None:
        attention_type = (
            "gqa"
            if runtime_config.get("kv_heads") is not None
            else "mha"
        )

    artifacts_dir = artifacts_root / attention_type
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    training_config_path = (
        artifacts_dir / "training_config.json"
    )

    best_model_path = artifacts_dir / "best_model.pt"

    training_metadata = {
        "seq_len": seq_len,
        "stride": stride,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "accum_steps": accum_steps,
        "d_model": runtime_config["d_model"],
        "layers": runtime_config["layers"],
        "heads": runtime_config["heads"],
        "dropout": runtime_config["dropout"],
        "d_ff": runtime_config["d_ff"],
        "kv_heads": runtime_config.get("kv_heads"),
        "vocab_size": vocab_size,
        "pad_id": pad_id,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "device": DEVICE.type,
        "smoke_test": args.smoke_test,
        "max_train_tokens": runtime_config["max_train_tokens"],
        "max_steps_per_epoch": runtime_config["max_steps_per_epoch"],
        "sample_every_epochs": sample_every_epochs,
        "attention_type": attention_type,
        "attention_backend": runtime_config.get(
            "attention_backend", "sdpa"
        ),
        "position_encoding": runtime_config.get(
            "position_encoding", "sinusoidal"
        ),
        "rope_theta": runtime_config.get(
            "rope_theta", 10000.0
        ),
        "rotary_dim": runtime_config.get("rotary_dim"),
        "mla": (
            runtime_config.get("mla")
            if attention_type == "mla"
            else None
        ),
    }

    with training_config_path.open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(training_metadata, f, indent=2)

    print("VOCAB_SIZE:", vocab_size)
    print("Max token ID:", max_token_id)
    print(f"Training config saved to: {TRAINING_CONFIG_FILE}")

    split = int(0.9 * len(tokens))
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    train_dataset = TokenDataset(train_tokens, seq_len, stride)
    val_dataset = TokenDataset(val_tokens, seq_len, stride)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=AMP_ENABLED,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=AMP_ENABLED,
    )

    model = build_transformer_from_config(
        training_metadata,
        tgt_vocab_size=vocab_size,
    ).to(DEVICE)

    print("Model parameters:", sum(p.numel() for p in model.parameters()))

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)
    tokenizer = load_tokenizer()

    global_step = 0
    optimizer_step = 0
    best_val_loss = float("inf")

    x, y = next(iter(train_loader))
    print("Input shape:", x.shape)
    print("Target shape:", y.shape)
    print(f"Dataset stride: {stride}")

    if args.smoke_test:
        print("Smoke test mode is enabled.")
        print(f"Max train tokens: {runtime_config['max_train_tokens']}")
        print(f"Max steps per epoch: {runtime_config['max_steps_per_epoch']}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        steps_completed = 0
        pending_microbatches = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        optimizer.zero_grad()

        for step, (x, y) in enumerate(progress_bar):
            if runtime_config["max_steps_per_epoch"] is not None and step >= runtime_config["max_steps_per_epoch"]:
                break

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            current_lr = get_lr(optimizer_step, lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            tgt_mask = model.create_tgt_mask(x, pad_id)

            with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
                logits = model(x, tgt_mask)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                loss = loss / accum_steps

            scaler.scale(loss).backward()
            pending_microbatches += 1

            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                optimizer_step += 1
                pending_microbatches = 0

            total_loss += loss.item() * accum_steps
            tokens_seen = (global_step + 1) * batch_size * seq_len
            steps_completed += 1

            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item() * accum_steps:.4f}",
                    "lr": f"{current_lr:.6f}",
                    "tokens": tokens_seen,
                }
            )

            global_step += 1

        if pending_microbatches > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            optimizer_step += 1

        avg_train_loss = total_loss / max(steps_completed, 1)

        model.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for val_step, (x, y) in enumerate(val_loader):
                if runtime_config["max_steps_per_epoch"] is not None and val_step >= runtime_config["max_steps_per_epoch"]:
                    break

                x = x.to(DEVICE)
                y = y.to(DEVICE)
                tgt_mask = model.create_tgt_mask(x, pad_id)
                logits = model(x, tgt_mask)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                val_loss += loss.item()
                val_steps += 1

        avg_val_loss = val_loss / max(val_steps, 1)
        perplexity = math.exp(avg_val_loss)

        print(f"\nEpoch {epoch + 1}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f}")
        print(f"Perplexity: {perplexity:.2f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), BEST_MODEL_FILE)
            print(f"Best model saved to {BEST_MODEL_FILE}")

        checkpoint_path = artifacts_dir / f"model_epoch_{epoch + 1}.pt"
        torch.save(
            model.state_dict(),
            best_model_path,
        )
        print(f"Checkpoint saved to {checkpoint_path}")

        if (epoch + 1) % sample_every_epochs == 0:
            run_sample_generations(
                model=model,
                tokenizer=tokenizer,
                seq_len=seq_len,
                pad_id=pad_id,
                eos_id=eos_id,
                bos_id=bos_id,
                device=DEVICE,
            )


if __name__ == "__main__":
    main()
