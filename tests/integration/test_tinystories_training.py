
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models

from latent_forge.models.decoder import (
    build_transformer_from_config,
)
from latent_forge.training.train_tinystories import (
    main,
    parse_args,
    sha256_file,
)


@pytest.mark.parametrize(
    "attention",
    ["mha", "gqa", "mqa", "mla"],
)
def test_phase2_smoke_all_attention_types(
    tmp_path: Path,
    attention: str,
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Minimal tokenizer fixture.
    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
    }
    vocab.update({
        f"word{i}": i for i in range(4, 12)
    })

    tokenizer = Tokenizer(
        models.WordLevel(
            vocab=vocab,
            unk_token="<unk>",
        )
    )

    tokenizer.save(str(data_dir / "tokenizer.json"))

    # Small packed token streams for testing.
    tokens = (
        (np.arange(640) % 8) + 4
    ).astype(np.uint16)

    tokens.tofile(data_dir / "train.bin")

    tokens[:320].tofile(
        data_dir / "validation.bin"
    )

    manifest = {
        "vocab_size": len(vocab),
        "pad_id": 0,
        "bos_id": 1,
        "eos_id": 2,
        "dtype": "uint16",
        "tokenizer_sha256": sha256_file(
            data_dir / "tokenizer.json"
        ),
        "train": {
            "sha256": sha256_file(
                data_dir / "train.bin"
            )
        },
        "validation": {
            "sha256": sha256_file(
                data_dir / "validation.bin"
            )
        },
    }

    (data_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    args = parse_args([
        "--attention", attention,
        "--data-dir", str(data_dir),
        "--output-dir", str(tmp_path / "runs"),
        "--mode", "overfit",
        "--steps", "2",
        "--warmup-steps", "0",
        "--eval-every", "1",
        "--val-batches", "1",
        "--seq-len", "16",
        "--d-model", "32",
        "--layers", "2",
        "--heads", "4",
        "--d-ff", "64",
        "--batch-size", "2",
        "--accum-steps", "1",
        "--overfit-windows", "2",
        "--device", "cpu",
        "--precision", "fp32",
        "--mla-q-rank", "16",
        "--mla-kv-rank", "16",
        "--mla-rope-dim", "8",
    ])

    run_dir = main(args)

    # Check the experiment bundle.
    assert (run_dir / "best_model.pt").is_file()
    assert (run_dir / "last_model.pt").is_file()
    assert (run_dir / "tokenizer.json").is_file()
    assert (run_dir / "results.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()

    # Reconstruct architecture from saved metadata.
    config = json.loads(
        (run_dir / "training_config.json").read_text()
    )

    model = build_transformer_from_config(
        config,
        tgt_vocab_size=config["vocab_size"],
    ).eval()

    model.load_state_dict(
        torch.load(
            run_dir / "best_model.pt",
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )

    with torch.inference_mode():
        output = model(
            torch.randint(4, 12, (1, 8))
        )

    assert output.shape == (1, 8, len(vocab))
    assert torch.isfinite(output).all()
