
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import DynamicCache

from latent_forge.configs.mla import MLAConfig
from latent_forge.models.decoder import (
    FeedForward,
    LayerNorm,
    SwiGLUFeedForward,
    build_transformer,
    build_transformer_from_config,
)


def create_model(attention_type="gqa"):
    common = dict(
        tgt_vocab_size=101,
        tgt_seq_len=32,
        d_model=64,
        N=3,
        h=4,
        dropout=0.0,
        d_ff=96,
        position_encoding="rope",
        norm_type="rmsnorm",
        ffn_type="swiglu",
    )

    if attention_type == "mla":
        common.update(
            attention_type="mla",
            kv_heads=None,
            mla_config=MLAConfig(
                hidden_size=64,
                num_attention_heads=4,
                kv_lora_rank=24,
                qk_nope_head_dim=12,
                qk_rope_head_dim=8,
                v_head_dim=16,
                q_lora_rank=32,
                implementation="absorbed",
            ),
        )
    else:
        common.update(
            attention_type="gqa",
            kv_heads=2,
        )

    return build_transformer(**common)


def test_rmsnorm_matches_manual_math():
    torch.manual_seed(42)

    x = torch.randn(2, 5, 64)

    norm = nn.RMSNorm(64, eps=1e-6)

    actual = norm(x)

    rms = torch.rsqrt(
        x.square().mean(dim=-1, keepdim=True)
        + 1e-6
    )

    expected = x * rms * norm.weight

    torch.testing.assert_close(
        actual,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


def test_swiglu_matches_manual_math():
    torch.manual_seed(42)

    ffn = SwiGLUFeedForward(
        d_model=64,
        d_ff=96,
        dropout=0.0,
    ).eval()

    x = torch.randn(2, 5, 64)

    actual = ffn(x)

    gate = F.silu(ffn.gate_proj(x))
    up = ffn.up_proj(x)

    expected = ffn.down_proj(gate * up)

    torch.testing.assert_close(
        actual,
        expected,
    )

    assert actual.shape == x.shape


@pytest.mark.parametrize(
    "attention_type",
    ["gqa", "mla"],
)
def test_modern_decoder_layers(attention_type):
    model = create_model(attention_type)

    assert isinstance(
        model.decoder.norm,
        nn.RMSNorm,
    )

    assert len(model.decoder.layers) == 3

    for block in model.decoder.layers:

        assert isinstance(
            block.feed_forward_block,
            SwiGLUFeedForward,
        )

        for residual in block.residual_connections:
            assert isinstance(
                residual.norm,
                nn.RMSNorm,
            )


@pytest.mark.parametrize(
    "attention_type",
    ["gqa", "mla"],
)
def test_cached_logits_match_full(attention_type):
    torch.manual_seed(42)

    model = create_model(attention_type).eval()

    tokens = torch.randint(
        1, 101, (2, 11)
    )

    with torch.inference_mode():

        full_logits = model(tokens)

        cache = DynamicCache()
        outputs = []

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

        cached_logits = torch.cat(
            outputs,
            dim=1,
        )

    torch.testing.assert_close(
        full_logits,
        cached_logits,
        atol=1e-4,
        rtol=1e-4,
    )

    for layer_idx in range(3):
        assert cache.get_seq_length(layer_idx) == 11

    if attention_type == "mla":
        assert cache[0][0].shape == (
            2, 1, 11, 24
        )
        assert cache[0][1].shape == (
            2, 1, 11, 8
        )
    else:
        assert cache[0][0].shape == (
            2, 2, 11, 16
        )


@pytest.mark.parametrize(
    "attention_type",
    ["gqa", "mla"],
)
def test_training_gradients(attention_type):
    torch.manual_seed(42)

    model = create_model(attention_type).train()

    tokens = torch.randint(
        1, 101, (2, 10)
    )

    logits = model(tokens)

    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, 101),
        tokens[:, 1:].reshape(-1),
    )

    loss.backward()

    for block in model.decoder.layers:

        ffn = block.feed_forward_block

        for projection in (
            ffn.gate_proj,
            ffn.up_proj,
            ffn.down_proj,
        ):
            grad = projection.weight.grad

            assert grad is not None
            assert torch.isfinite(grad).all()

        for residual in block.residual_connections:
            grad = residual.norm.weight.grad

            assert grad is not None
            assert torch.isfinite(grad).all()


def test_old_architecture_remains_available():
    model = build_transformer(
        tgt_vocab_size=101,
        tgt_seq_len=32,
        d_model=64,
        N=2,
        h=4,
        kv_heads=2,
        dropout=0.0,
        d_ff=96,
    )

    assert isinstance(
        model.decoder.norm,
        LayerNorm,
    )

    for block in model.decoder.layers:
        assert isinstance(
            block.feed_forward_block,
            FeedForward,
        )


def test_model_reconstruction_from_metadata():
    metadata = {
        "vocab_size": 101,
        "seq_len": 32,
        "d_model": 64,
        "layers": 2,
        "heads": 4,
        "kv_heads": 2,
        "dropout": 0.0,
        "d_ff": 96,
        "attention_type": "gqa",
        "position_encoding": "rope",
        "norm_type": "rmsnorm",
        "ffn_type": "swiglu",
    }

    model = build_transformer_from_config(
        metadata,
        tgt_vocab_size=101,
    )

    assert isinstance(
        model.decoder.norm,
        nn.RMSNorm,
    )

    assert isinstance(
        model.decoder.layers[0].feed_forward_block,
        SwiGLUFeedForward,
    )


def test_invalid_options():
    kwargs = dict(
        tgt_vocab_size=101,
        tgt_seq_len=32,
        d_model=64,
        N=2,
        h=4,
        kv_heads=2,
        dropout=0.0,
    )

    with pytest.raises(ValueError):
        build_transformer(
            **kwargs,
            norm_type="invalid",
        )

    with pytest.raises(ValueError):
        build_transformer(
            **kwargs,
            ffn_type="invalid",
        )
