
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class TinyStoriesTokenizer:
    """
    Adapter exposing LatentForge's encode/decode API.
    """

    def __init__(self, path: str | Path):
        self.tokenizer = Tokenizer.from_file(str(path))

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(
            text,
            add_special_tokens=False,
        ).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(
            ids,
            skip_special_tokens=True,
        )

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()


class PackedTokenDataset(Dataset):
    """
    Next-token prediction on a packed token stream.

    x = tokens[start : start + seq_len]
    y = tokens[start + 1 : start + seq_len + 1]

    No train/validation split is performed here.
    Each instance reads exactly one prepared split.
    """

    def __init__(
        self,
        path: str | Path,
        seq_len: int,
        stride: int | None = None,
        dtype: str = "uint16",
    ):
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")

        if stride is None:
            stride = seq_len

        if stride <= 0:
            raise ValueError("stride must be positive")

        self.tokens = np.memmap(
            path,
            mode="r",
            dtype=np.dtype(dtype),
        )

        self.seq_len = seq_len
        self.stride = stride

        if len(self.tokens) <= seq_len:
            raise ValueError(
                "Token stream is shorter than the "
                "requested sequence length"
            )

        self.num_windows = (
            (len(self.tokens) - seq_len - 1) // stride
        ) + 1

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        start = idx * self.stride
        end = start + self.seq_len + 1

        # Copy into a writable int64 buffer.
        window = np.array(
            self.tokens[start:end],
            dtype=np.int64,
            copy=True,
        )

        x = torch.from_numpy(window[:-1].copy())
        y = torch.from_numpy(window[1:].copy())

        return x, y


def load_tinystories_data(
    root: str | Path,
    seq_len: int = 256,
):
    root = Path(root)

    with (root / "manifest.json").open(
        encoding="utf-8"
    ) as f:
        manifest = json.load(f)

    tokenizer = TinyStoriesTokenizer(
        root / "tokenizer.json"
    )

    if tokenizer.vocab_size != manifest["vocab_size"]:
        raise ValueError(
            "Tokenizer vocabulary differs from manifest"
        )

    train_dataset = PackedTokenDataset(
        root / "train.bin",
        seq_len=seq_len,
        stride=seq_len,
        dtype=manifest["dtype"],
    )

    val_dataset = PackedTokenDataset(
        root / "validation.bin",
        seq_len=seq_len,
        stride=seq_len,
        dtype=manifest["dtype"],
    )

    return train_dataset, val_dataset, tokenizer, manifest
