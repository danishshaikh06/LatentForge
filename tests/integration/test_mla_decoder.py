
import io

import pytest
import torch
from transformers.cache_utils import DynamicCache

from latent_forge.configs.mla import MLAConfig
from latent_forge.models.attention.mla import (
    MultiHeadLatentAttention,
)
from latent_forge.models.decoder import (
    build_transformer,
    build_transformer_from_config,
)


def make_mla_model(mode="naive"):
    cfg = MLAConfig(
        hidden_size=64,
        num_attention_heads=4,
        kv_lora_rank=24,
        qk_nope_head_dim=12,
        qk_rope_head_dim=8,
        v_head_dim=16,
        q_lora_rank=32,
        attention_dropout=0.0,
        implementation=mode,
    )

    return build_transformer(
        tgt_vocab_size=97,
        tgt_seq_len=32,
        d_model=64,
        N=3,
        h=4,
        dropout=0.0,
        d_ff=128,
        kv_heads=None,
        attention_type="mla",
        position_encoding="rope",
        mla_config=cfg,
    )


def test_all_decoder_layers_use_mla():
    model = make_mla_model()

    assert model.attention_type == "mla"
    assert model.rotary_emb is None
    assert model.tgt_pos is None

    for idx, layer in enumerate(model.decoder.layers):
        attention = layer.self_attention_block

        assert isinstance(
            attention, MultiHeadLatentAttention
        )
        assert attention.layer_idx == idx


@pytest.mark.parametrize("mode", ["naive", "absorbed"])
def test_full_vs_cached_logits(mode):
    torch.manual_seed(42)

    model = make_mla_model(mode).eval()
    tokens = torch.randint(1, 97, (2, 11))

    with torch.inference_mode():
        full_logits = model(tokens)

        cache = DynamicCache()
        outputs = []

        # Prefill + incremental + multi-token decoding.
        for start, end in (
            (0, 4),
            (4, 5),
            (5, 8),
            (8, 11),
        ):
            logits, returned_cache = model(
                tokens[:, start:end],
                past_key_values=cache,
                use_cache=True,
            )

            assert returned_cache is cache
            outputs.append(logits)

        cached_logits = torch.cat(outputs, dim=1)

    torch.testing.assert_close(
        full_logits,
        cached_logits,
        atol=1e-4,
        rtol=1e-4,
    )

    for layer_idx in range(3):
        assert cache.get_seq_length(layer_idx) == 11

        # Compressed latent, not expanded keys.
        assert cache[layer_idx][0].shape == (
            2, 1, 11, 24
        )

        # Shared rotated positional key.
        assert cache[layer_idx][1].shape == (
            2, 1, 11, 8
        )


def test_naive_matches_absorbed_end_to_end():
    torch.manual_seed(42)

    naive = make_mla_model("naive").eval()
    absorbed = make_mla_model("absorbed").eval()

    # Same learned parameters, different computation.
    absorbed.load_state_dict(naive.state_dict())

    tokens = torch.randint(1, 97, (2, 9))

    with torch.inference_mode():
        naive_logits = naive(tokens)
        absorbed_logits = absorbed(tokens)

    torch.testing.assert_close(
        naive_logits,
        absorbed_logits,
        atol=1e-4,
        rtol=1e-4,
    )


def test_batched_padding_mask_with_cache():
    model = make_mla_model("absorbed").eval()

    tokens = torch.tensor([
        [0, 0, 11, 12, 13, 14, 15],
        [21, 22, 23, 24, 25, 26, 27],
    ])

    key_mask = tokens.ne(0)

    with torch.inference_mode():
        full_mask = model.create_tgt_mask(
            tokens,
            pad_token_id=0,
        )

        full = model(tokens, full_mask)

        cache = DynamicCache()
        pieces = []

        for start, end in ((0, 3), (3, 4), (4, 7)):
            logits, _ = model(
                tokens[:, start:end],
                key_mask[:, :end],
                past_key_values=cache,
                use_cache=True,
            )

            pieces.append(logits)

        cached = torch.cat(pieces, dim=1)

    torch.testing.assert_close(
        full,
        cached,
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.parametrize("mode", ["naive", "absorbed"])
def test_full_model_backward(mode):
    torch.manual_seed(42)

    model = make_mla_model(mode).train()

    tokens = torch.randint(1, 97, (2, 10))

    logits = model(tokens)

    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 97),
        tokens[:, 1:].reshape(-1),
    )

    loss.backward()

    for layer in model.decoder.layers:
        attention = layer.self_attention_block

        projections = (
            attention.q_a_proj,
            attention.q_b_proj,
            attention.kv_a_proj,
            attention.kv_b_proj,
            attention.o_proj,
        )

        for proj in projections:
            assert proj.weight.grad is not None

            assert torch.isfinite(
                proj.weight.grad
            ).all()


def test_checkpoint_roundtrip():
    torch.manual_seed(42)

    first = make_mla_model().eval()
    second = make_mla_model().eval()

    # No filesystem artifact needed for this test.
    buffer = io.BytesIO()
    torch.save(first.state_dict(), buffer)
    buffer.seek(0)

    second.load_state_dict(
        torch.load(buffer, weights_only=True),
        strict=True,
    )

    tokens = torch.randint(1, 97, (1, 8))

    with torch.inference_mode():
        expected = first(tokens)
        actual = second(tokens)

    torch.testing.assert_close(actual, expected)


def test_model_creation_from_metadata():
    metadata = {
        "vocab_size": 97,
        "seq_len": 32,
        "d_model": 64,
        "layers": 3,
        "heads": 4,
        "dropout": 0.0,
        "d_ff": 128,
        "kv_heads": None,
        "attention_type": "mla",
        "position_encoding": "rope",
        "mla": {
            "q_lora_rank": 32,
            "kv_lora_rank": 24,
            "qk_nope_head_dim": 12,
            "qk_rope_head_dim": 8,
            "v_head_dim": 16,
            "implementation": "absorbed",
        },
    }

    model = build_transformer_from_config(
        metadata,
        tgt_vocab_size=metadata["vocab_size"],
    )

    assert model.attention_type == "mla"

    attention = model.decoder.layers[0].self_attention_block

    assert attention.config.implementation == "absorbed"
    assert attention.config.kv_lora_rank == 24


def test_legacy_gqa_still_constructs():
    model = build_transformer(
        tgt_vocab_size=97,
        tgt_seq_len=32,
        d_model=64,
        N=2,
        h=4,
        kv_heads=2,
        position_encoding="rope",
        dropout=0.0,
        d_ff=128,
    ).eval()

    assert model.attention_type == "gqa"

    with torch.inference_mode():
        logits = model(torch.randint(1, 97, (1, 6)))

    assert logits.shape == (1, 6, 97)


def test_invalid_mla_configuration_rejected():
    with pytest.raises(ValueError):
        build_transformer(
            tgt_vocab_size=97,
            tgt_seq_len=32,
            d_model=64,
            N=2,
            h=4,
            dropout=0.0,
            attention_type="mla",
            position_encoding="rope",
            mla_config=None,
        )


def test_request_cache_isolation():
    model = make_mla_model().eval()

    cache_a = DynamicCache()
    cache_b = DynamicCache()

    with torch.inference_mode():
        model(
            torch.randint(1, 97, (1, 5)),
            past_key_values=cache_a,
            use_cache=True,
        )

        model(
            torch.randint(1, 97, (1, 2)),
            past_key_values=cache_b,
            use_cache=True,
        )

    for layer_idx in range(3):
        assert cache_a.get_seq_length(layer_idx) == 5
        assert cache_b.get_seq_length(layer_idx) == 2
