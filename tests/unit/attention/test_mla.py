import pytest
import torch
from transformers.cache_utils import DynamicCache

from latent_forge.configs.mla import MLAConfig
from latent_forge.models.attention.mla import MultiHeadLatentAttention


def make_module(mode="naive", q_lora_rank=32, layer_idx=0):
    cfg = MLAConfig(
        hidden_size=64,
        num_attention_heads=4,
        kv_lora_rank=24,
        qk_nope_head_dim=12,
        qk_rope_head_dim=8,
        v_head_dim=16,
        q_lora_rank=q_lora_rank,
        attention_dropout=0,
        implementation=mode,
    )
    return MultiHeadLatentAttention(cfg, layer_idx=layer_idx)


@pytest.mark.parametrize("q_rank", [None, 32])
def test_naive_and_absorbed_match(q_rank):
    torch.manual_seed(7)
    m = make_module(q_lora_rank=q_rank).eval()
    x = torch.randn(2, 9, 64)
    mask = torch.tensor([[True]*9, [False, False] + [True]*7])
    with torch.inference_mode():
        naive = m(x, attention_mask=mask, implementation="naive")
        absorbed = m(x, attention_mask=mask, implementation="absorbed")
    torch.testing.assert_close(naive, absorbed, rtol=1e-5, atol=1e-6)
    assert naive.shape == (2, 9, 64)


@pytest.mark.parametrize("mode", ["naive", "absorbed"])
@pytest.mark.parametrize("q_rank", [None, 32])
def test_cached_matches_full(mode, q_rank):
    torch.manual_seed(8)
    m = make_module(q_lora_rank=q_rank).eval()
    x = torch.randn(2, 11, 64)
    with torch.no_grad():
        full = m(x, implementation=mode)
        cache = DynamicCache()
        pieces = []
        for lo, hi in [(0, 4), (4, 5), (5, 8), (8, 11)]:
            pieces.append(
                m(x[:, lo:hi], past_key_values=cache,
                  use_cache=True, implementation=mode)
            )
        cached = torch.cat(pieces, dim=1)
    torch.testing.assert_close(full, cached, rtol=1e-5, atol=1e-6)
    assert cache.get_seq_length(0) == 11
    assert cache[0][0].shape == (2, 1, 11, 24)
    assert cache[0][1].shape == (2, 1, 11, 8)


def test_naive_prefill_then_absorbed_decode():
    torch.manual_seed(9)
    m = make_module().eval()
    x = torch.randn(2, 9, 64)
    with torch.no_grad():
        expected = m(x, implementation="naive")
        cache = DynamicCache()
        first = m(x[:, :5], past_key_values=cache, use_cache=True, implementation="naive")
        last = m(x[:, 5:], past_key_values=cache, use_cache=True, implementation="absorbed")
    torch.testing.assert_close(torch.cat([first, last], dim=1), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("mode", ["naive", "absorbed"])
def test_backward(mode):
    torch.manual_seed(10)
    m = make_module().train()
    x = torch.randn(2, 7, 64, requires_grad=True)
    out = m(x, implementation=mode)
    loss = out.square().mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for n, p in m.named_parameters():
        assert p.grad is not None, n
        assert torch.isfinite(p.grad).all(), n


@pytest.mark.parametrize("mode", ["naive", "absorbed"])
def test_fully_masked_row(mode):
    torch.manual_seed(11)
    m = make_module().eval()
    x = torch.randn(2, 5, 64)
    mask = torch.tensor([[False]*5, [True]*5])
    with torch.no_grad():
        out = m(x, attention_mask=mask, implementation=mode)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[0], torch.zeros_like(out[0]))


def test_invalid_mask_does_not_mutate_cache():
    m = make_module().eval()
    cache = DynamicCache()
    with pytest.raises(ValueError):
        m(torch.randn(1, 3, 64), attention_mask=torch.ones(1, 4, dtype=torch.bool),
          past_key_values=cache, use_cache=True)
    assert cache.get_seq_length(0) == 0


def test_independent_layer_cache():
    m0, m1 = make_module(layer_idx=0), make_module(layer_idx=1)
    cache = DynamicCache()
    x = torch.randn(1, 4, 64)
    m0(x, past_key_values=cache, use_cache=True)
    m1(x, past_key_values=cache, use_cache=True)
    assert cache.get_seq_length(0) == cache.get_seq_length(1) == 4


def test_config_validation():
    with pytest.raises(ValueError):
        MLAConfig(hidden_size=64, num_attention_heads=4, kv_lora_rank=24,
                  qk_nope_head_dim=12, qk_rope_head_dim=7, v_head_dim=16)


def test_naive_absorbed_gradients_match():
    torch.manual_seed(15)
    a, b = make_module().train(), make_module().train()
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 6, 64)
    # Use same deterministic forward; dropout is disabled in configuration.
    ya = a(x, implementation="naive").square().mean()
    yb = b(x, implementation="absorbed").square().mean()
    ya.backward()
    yb.backward()
    for (name_a, pa), (name_b, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert name_a == name_b
        torch.testing.assert_close(pa.grad, pb.grad, atol=2e-6, rtol=2e-5)


def test_request_cache_isolation():
    m = make_module().eval()
    a, b = DynamicCache(), DynamicCache()
    m(torch.randn(1, 2, 64), past_key_values=a, use_cache=True)
    m(torch.randn(1, 5, 64), past_key_values=b, use_cache=True)
    assert a.get_seq_length(0) == 2
    assert b.get_seq_length(0) == 5


def dense_reference(module, x, mask=None):
    """Explicit dense MLA equations, without calling attention or RoPE helpers."""
    cfg = module.config
    batch, length, _ = x.shape
    heads = cfg.num_attention_heads
    if module.q_proj is not None:
        query = torch.nn.functional.linear(x, module.q_proj.weight)
    else:
        query = torch.nn.functional.linear(x, module.q_a_proj.weight)
        query = query * torch.rsqrt(query.square().mean(-1, keepdim=True) + 1e-6)
        query = query * module.q_a_norm.weight
        query = torch.nn.functional.linear(query, module.q_b_proj.weight)
    query = query.reshape(batch, length, heads, cfg.qk_head_dim).transpose(1, 2)
    q_content, q_position = query.split(
        [cfg.qk_nope_head_dim, cfg.qk_rope_head_dim], dim=-1
    )
    compressed = torch.nn.functional.linear(x, module.kv_a_proj.weight)
    latent, k_position = compressed.split([cfg.kv_lora_rank, cfg.qk_rope_head_dim], -1)
    latent = latent * torch.rsqrt(latent.square().mean(-1, keepdim=True) + 1e-6)
    latent = latent * module.kv_a_norm.weight
    kv = torch.nn.functional.linear(latent, module.kv_b_proj.weight)
    kv = kv.reshape(batch, length, heads, -1).transpose(1, 2)
    k_content, values = kv.split([cfg.qk_nope_head_dim, cfg.v_head_dim], -1)
    positions = torch.arange(length, device=x.device)
    frequencies = cfg.rope_theta ** (
        -torch.arange(0, cfg.qk_rope_head_dim, 2, device=x.device).float()
        / cfg.qk_rope_head_dim
    )
    angles = positions[:, None] * frequencies[None, :]
    angles = torch.cat([angles, angles], -1)[None, None]

    def rotate(tensor):
        left, right = tensor.chunk(2, -1)
        return tensor * angles.cos() + torch.cat([-right, left], -1) * angles.sin()

    query = torch.cat([q_content, rotate(q_position)], -1)
    keys = torch.cat([
        k_content, rotate(k_position.unsqueeze(1)).expand(-1, heads, -1, -1)
    ], -1)
    scores = query @ keys.transpose(-1, -2) / cfg.qk_head_dim ** 0.5
    allowed = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
    allowed = allowed[None, None].expand(batch, heads, -1, -1)
    if mask is not None:
        allowed = allowed & (mask[:, None, None, :] if mask.ndim == 2 else mask)
    scores = scores.float().masked_fill(~allowed, float('-inf'))
    # Explicitly define empty rows as zero without applying softmax to -inf rows.
    nonempty = allowed.any(-1, keepdim=True)
    probabilities = scores.masked_fill(~nonempty, 0).softmax(-1)
    probabilities = probabilities.masked_fill(~allowed, 0)
    output = (probabilities.to(values.dtype) @ values).transpose(1, 2)
    return torch.nn.functional.linear(
        output.reshape(batch, length, heads * cfg.v_head_dim), module.o_proj.weight
    )


@pytest.mark.parametrize('mode', ['naive', 'absorbed'])
@pytest.mark.parametrize('q_rank', [None, 32])
@pytest.mark.parametrize('mask_kind', ['none', 'padding', 'shared', 'heads'])
def test_matches_independent_dense_reference(mode, q_rank, mask_kind):
    torch.manual_seed(21)
    module = make_module(q_lora_rank=q_rank).eval()
    x = torch.randn(2, 5, 64)
    mask = None
    if mask_kind == 'padding':
        mask = torch.tensor([[False, False, True, True, True], [True] * 5])
    elif mask_kind in ('shared', 'heads'):
        heads = 1 if mask_kind == 'shared' else 4
        mask = torch.rand(2, heads, 5, 5) > 0.3
        mask[:, :, 2, :] = False
    with torch.no_grad():
        actual = module(x, attention_mask=mask, implementation=mode)
        expected = dense_reference(module, x, mask)
    assert actual.shape == x.shape
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('mode', ['naive', 'absorbed'])
def test_causality_and_masked_keys(mode):
    torch.manual_seed(22)
    module = make_module().eval()
    x = torch.randn(2, 6, 64)
    changed = x.clone()
    changed[:, 3:] += torch.randn_like(changed[:, 3:]) * 10
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[:, 1] = False
    masked_change = x.clone()
    masked_change[:, 1] += 100
    with torch.no_grad():
        original = module(x, implementation=mode)
        future_changed = module(changed, implementation=mode)
        masked = module(x, attention_mask=mask, implementation=mode)
        masked_changed = module(masked_change, attention_mask=mask, implementation=mode)
    torch.testing.assert_close(original[:, :3], future_changed[:, :3])
    # The changed token's own query changes, but other queries must be unaffected.
    torch.testing.assert_close(masked[:, [0, 2, 3, 4, 5]],
                               masked_changed[:, [0, 2, 3, 4, 5]])


@pytest.mark.parametrize('mode', ['naive', 'absorbed'])
@pytest.mark.parametrize('q_rank', [None, 32])
def test_single_token_and_zero_input(mode, q_rank):
    module = make_module(q_lora_rank=q_rank).eval()
    x = torch.randn(1, 1, 64)
    with torch.no_grad():
        output = module(x, implementation=mode)
        # With one key, attention probability is exactly one for every head.
        latent = module.kv_a_norm(module.kv_a_proj(x)[..., :24])
        values = module.kv_b_proj(latent).reshape(1, 1, 4, 28)[..., 12:]
        expected = module.o_proj(values.reshape(1, 1, 64))
        zeros = module(torch.zeros(1, 3, 64), implementation=mode)
    torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(zeros, torch.zeros_like(zeros), atol=0, rtol=0)


@pytest.mark.parametrize('mode', ['naive', 'absorbed'])
def test_masked_cached_decode_with_explicit_positions(mode):
    torch.manual_seed(23)
    module = make_module().eval()
    x = torch.randn(2, 7, 64)
    mask = torch.tensor([[False, True, True, True, True, False, True], [True] * 7])
    cache = DynamicCache()
    with torch.no_grad():
        expected = module(x, attention_mask=mask, implementation=mode)
        pieces = []
        for start, end in [(0, 3), (3, 4), (4, 7)]:
            pieces.append(module(
                x[:, start:end], attention_mask=mask[:, :end],
                cache_position=torch.arange(start, end), past_key_values=cache,
                use_cache=True, implementation=mode,
            ))
    torch.testing.assert_close(torch.cat(pieces, 1), expected, atol=1e-6, rtol=1e-5)
    assert cache[0][0].shape == (2, 1, 7, 24)
    assert cache[0][1].shape == (2, 1, 7, 8)


@pytest.mark.parametrize('shape', [(2, 64), (2, 3, 63), (0, 3, 64), (2, 0, 64)])
def test_rejects_invalid_hidden_shape(shape):
    with pytest.raises(ValueError, match='shape|Empty'):
        make_module()(torch.randn(shape))


def test_rejects_integer_hidden_states():
    with pytest.raises(TypeError, match='floating point'):
        make_module()(torch.ones(1, 2, 64, dtype=torch.long))


@pytest.mark.parametrize('case', ['mask_shape', 'mask_dtype', 'position_shape',
                                  'position_dtype', 'mode'])
def test_invalid_arguments_preserve_existing_cache(case):
    module = make_module().eval()
    cache = DynamicCache()
    with torch.no_grad():
        module(torch.randn(1, 2, 64), past_key_values=cache, use_cache=True)
    before = tuple(t.clone() for t in cache[0])
    kwargs = {
        'mask_shape': {'attention_mask': torch.ones(1, 3, dtype=torch.bool)},
        'mask_dtype': {'attention_mask': torch.ones(1, 4)},
        'position_shape': {'cache_position': torch.tensor([[2, 3]])},
        'position_dtype': {'cache_position': torch.tensor([2., 3.])},
        'mode': {'implementation': 'unknown'},
    }[case]
    with pytest.raises(ValueError):
        module(torch.randn(1, 2, 64), past_key_values=cache, use_cache=True, **kwargs)
    assert cache.get_seq_length(0) == 2
    for actual, expected in zip(cache[0], before):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize('case', ['missing', 'wrong_type', 'disabled'])
def test_cache_contract(case):
    kwargs = {
        'missing': {'use_cache': True},
        'wrong_type': {'use_cache': True, 'past_key_values': object()},
        'disabled': {'past_key_values': DynamicCache()},
    }[case]
    with pytest.raises(ValueError, match='Cache'):
        make_module()(torch.randn(1, 2, 64), **kwargs)


@pytest.mark.parametrize('layer_idx', [-1, True, 0.5])
def test_invalid_layer_index(layer_idx):
    with pytest.raises(ValueError, match='layer_idx'):
        make_module(layer_idx=layer_idx)


@pytest.mark.parametrize('mode', ['naive', 'absorbed'])
def test_dropout_and_eval_determinism(mode):
    from dataclasses import replace

    torch.manual_seed(24)
    cfg = replace(make_module().config, attention_dropout=0.5, implementation=mode)
    module = MultiHeadLatentAttention(cfg, layer_idx=0)
    x = torch.randn(2, 6, 64)
    module.train()
    torch.manual_seed(25)
    first = module(x)
    torch.manual_seed(25)
    repeated = module(x)
    torch.manual_seed(26)
    different = module(x)
    torch.testing.assert_close(first, repeated, atol=0, rtol=0)
    assert not torch.allclose(first, different)
    module.eval()
    with torch.no_grad():
        evaluation = module(x)
        torch.testing.assert_close(module(x), evaluation, atol=0, rtol=0)
        torch.testing.assert_close(evaluation, dense_reference(module, x), atol=1e-6, rtol=1e-5)


# Exercise every available device; CUDA absence is reported by the run environment.
DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('mode', ['naive', 'absorbed'])
@pytest.mark.parametrize('q_rank', [None, 32])
def test_device_precision_and_backward(device, dtype, mode, q_rank):
    torch.manual_seed(27)
    reference = make_module(q_lora_rank=q_rank).eval().to(device)
    module = make_module(q_lora_rank=q_rank).eval().to(device)
    module.load_state_dict(reference.state_dict())
    module = module.to(dtype=dtype)
    x = torch.randn(2, 5, 64, device=device)
    mask = torch.tensor([[False] * 5, [True] * 5], device=device)
    with torch.no_grad():
        expected = reference(x, attention_mask=mask, implementation=mode)
    x = x.to(dtype).requires_grad_()
    output = module(x, attention_mask=mask, implementation=mode)
    assert output.shape == (2, 5, 64)
    assert output.device == x.device and output.dtype == dtype
    assert torch.isfinite(output).all()
    tolerance = {torch.float32: 1e-6, torch.float16: 3e-3, torch.bfloat16: 2e-2}[dtype]
    torch.testing.assert_close(output.float(), expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(output[0], torch.zeros_like(output[0]), atol=0, rtol=0)
    output.float().square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad[1].abs().sum() > 0
    torch.testing.assert_close(x.grad[0], torch.zeros_like(x.grad[0]), atol=0, rtol=0)
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize('q_rank', [None, 32])
def test_input_gradients_match_independent_reference(q_rank):
    torch.manual_seed(28)
    module = make_module(q_lora_rank=q_rank).eval()
    x = torch.randn(1, 3, 64, requires_grad=True)
    expected = dense_reference(module, x)
    expected_grads = torch.autograd.grad(expected.square().sum(), (x, *module.parameters()))
    for mode in ['naive', 'absorbed']:
        actual = module(x, implementation=mode)
        grads = torch.autograd.grad(actual.square().sum(), (x, *module.parameters()))
        for actual_grad, expected_grad in zip(grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, atol=3e-6, rtol=3e-5)
