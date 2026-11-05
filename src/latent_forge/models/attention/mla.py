"""DeepSeek-style MLA with naive and weight-absorbed attention implementations.

This module deliberately supports Hugging Face DynamicCache only. Its cache stores
normalized KV latents as keys and rotated shared positional keys as values.
The differing last dimensions are intentional and supported by DynamicCache.
"""
from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F
from transformers.cache_utils import Cache, DynamicCache

from latent_forge.configs.mla import MLAConfig
from latent_forge.models.attention.rope import RotaryEmbedding, apply_rotary_pos_emb


class MultiHeadLatentAttention(nn.Module):
    """Multi-head latent attention; one instance per Transformer decoder layer.

    Both paths cache (B, 1, T, rKV) and (B, 1, T, dR) instead of
    reconstructing and caching dense K and V.

    `naive`: reconstruct K/V for attention; good correctness/training reference.
    `absorbed`: aggregate in latent space, no dense cached K/V reconstruction.

    `past_key_values` is request-owned, mutated in place, and never placed in
    the module's parameters or persistent buffers.

    MLA TERMINOLOGY REFERENCE
    =========================

    The following notation maps the implementation's variable and
    module names to the mathematical terminology used in DeepSeek MLA.

    Tensor terminology
    ------------------

    Implementation name    MLA notation    Meaning
    ---------------------------------------------------------------
    hidden_states           x               Input representations
    q_nope                  q_C             Query content component
    q_rot                   q_R             Query RoPE component
    latent                  c_KV            Compressed KV latent
    k_rot                   k_R             Shared RoPE key component
    k_nope                  k_C             Key content component
    values                  v               Value representations

    Projection terminology
    ----------------------

    Implementation name    MLA notation    Meaning
    ---------------------------------------------------------------
    q_a_proj                W_DQ            Query down-projection
    q_b_proj                W_UQ            Query up-projection
    kv_a_proj               W_DKV + W_KR    KV down-projection and
                                            shared positional-key
                                            projection
    kv_b_proj               W_UK + W_UV     Key and value up-projections
    o_proj                  W_O             Output projection

    C = Content component (without RoPE)
    R = Rotary positional component
    D = Down-projection
    U = Up-projection
    KV = Key-Value
    O = Output projection
    """
    def __init__(self, config: MLAConfig, *, layer_idx: int) -> None:
        super().__init__()
        if isinstance(layer_idx, bool) or not isinstance(layer_idx, int) or layer_idx < 0:
            raise ValueError("layer_idx must be a nonnegative integer")

        self.config = config
        self.layer_idx = layer_idx
        H = config.num_attention_heads
        D = config.hidden_size
        d_qk = config.qk_head_dim

        if config.q_lora_rank is None:
            self.q_proj = nn.Linear(D, H * d_qk, bias=False)
            self.q_a_proj = None
            self.q_a_norm = None
            self.q_b_proj = None
        else:
            self.q_proj = None
            self.q_a_proj = nn.Linear(D, config.q_lora_rank, bias=False)
            self.q_a_norm = nn.RMSNorm(config.q_lora_rank, eps=1e-6)
            self.q_b_proj = nn.Linear(config.q_lora_rank, H * d_qk, bias=False)

        # Joint low-rank KV projection and shared positional key projection.
        self.kv_a_proj = nn.Linear(
            D, config.kv_lora_rank + config.qk_rope_head_dim, bias=False
        )
        self.kv_a_norm = nn.RMSNorm(config.kv_lora_rank, eps=1e-6)
        # Bias-free projection is required for exact weight absorption.
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            H * (config.qk_nope_head_dim + config.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(H * config.v_head_dim, D, bias=False)
        self.rotary_emb = RotaryEmbedding(config.qk_rope_head_dim, config.rope_theta)
        self.scale = config.qk_head_dim ** -0.5

    def _allowed_mask(
        self,
        *,
        batch: int,
        q_len: int,
        kv_len: int,
        positions: torch.Tensor,
        device: torch.device,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        keys = torch.arange(kv_len, device=device)
        allowed = (keys[None, :] <= positions[:, None])[None, None, :, :]

        if attention_mask is not None:
            if attention_mask.dtype != torch.bool or attention_mask.device != device:
                raise ValueError("attention_mask must be bool on hidden_states.device")
            H = self.config.num_attention_heads
            if attention_mask.shape == (batch, kv_len):
                attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.shape not in (
                (batch, 1, q_len, kv_len), (batch, H, q_len, kv_len)
            ):
                raise ValueError("Invalid attention_mask shape")
            allowed = allowed & attention_mask
        return allowed

    def _probabilities(self, scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Avoid NaNs on fully masked query rows, including in fp16/bfloat16.
        scores = scores.float().masked_fill(~mask, torch.finfo(torch.float32).min)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(mask.any(dim=-1, keepdim=True), weights, 0.0)
        return F.dropout(
            weights, p=self.config.attention_dropout, training=self.training
        )

    def _naive(
        self,
        q_nope: torch.Tensor,
        q_rot: torch.Tensor,
        latent: torch.Tensor,
        k_rot: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        B, _, T, _ = latent.shape
        H = self.config.num_attention_heads
        dc = self.config.qk_nope_head_dim
        dv = self.config.v_head_dim

        kv = self.kv_b_proj(latent.squeeze(1))
        kv = kv.reshape(B, T, H, dc + dv).transpose(1, 2)
        k_nope, values = kv.split((dc, dv), dim=-1)

        query = torch.cat((q_nope, q_rot), dim=-1)
        keys = torch.cat((k_nope, k_rot.expand(-1, H, -1, -1)), dim=-1)
        scores = (query @ keys.transpose(-2, -1)) * self.scale
        weights = self._probabilities(scores, mask)
        return weights.to(values.dtype) @ values

    def _absorbed(
        self,
        q_nope: torch.Tensor,
        q_rot: torch.Tensor,
        latent: torch.Tensor,
        k_rot: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        H = self.config.num_attention_heads
        dc = self.config.qk_nope_head_dim
        dv = self.config.v_head_dim
        r = self.config.kv_lora_rank
        # Linear.weight is (out_features, in_features), i.e. (H, dc+dv, r).
        w = self.kv_b_proj.weight.reshape(H, dc + dv, r)
        wk, wv = w.split((dc, dv), dim=1)
        c = latent.squeeze(1)  # (B, Tk, r)
        pe = k_rot.squeeze(1)  # (B, Tk, dR)

        # (B, H, Tq, dc) -> (B, H, Tq, r), absorb W_UK into Q.
        q_latent = torch.einsum("bhtd,hdr->bhtr", q_nope, wk)
        content_scores = torch.einsum("bhtr,bsr->bhts", q_latent, c)
        position_scores = torch.einsum("bhtd,bsd->bhts", q_rot, pe)
        scores = (content_scores + position_scores) * self.scale
        weights = self._probabilities(scores, mask)

        # Aggregate compressed values first; apply W_UV only to output tokens.
        value_latent = torch.einsum("bhts,bsr->bhtr", weights.to(c.dtype), c)
        return torch.einsum("bhtr,hdr->bhtd", value_latent, wv)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool = False,
        implementation: Literal["naive", "absorbed"] | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError("Expected hidden_states of shape (B, T, hidden_size)")
        if hidden_states.shape[0] == 0 or hidden_states.shape[1] == 0:
            raise ValueError("Empty batches/sequences are unsupported")
        if not hidden_states.is_floating_point():
            raise TypeError("hidden_states must be floating point")
        if use_cache and not isinstance(past_key_values, DynamicCache):
            raise ValueError("use_cache=True requires a request-owned DynamicCache")
        if not use_cache and past_key_values is not None:
            raise ValueError("Cache passed while use_cache=False")

        mode = implementation or self.config.implementation
        if mode not in ("naive", "absorbed"):
            raise ValueError("Invalid MLA implementation")

        B, T, _ = hidden_states.shape
        H = self.config.num_attention_heads
        dc = self.config.qk_nope_head_dim
        dr = self.config.qk_rope_head_dim
        past_len = (
            past_key_values.get_seq_length(self.layer_idx) if use_cache else 0
        )
        kv_len = past_len + T

        if cache_position is None:
            cache_position = torch.arange(
                past_len, kv_len, dtype=torch.long, device=hidden_states.device
            )
        elif (
            cache_position.shape != (T,)
            or cache_position.dtype != torch.long
            or cache_position.device != hidden_states.device
        ):
            raise ValueError("Invalid cache_position; expected (T,) int64 on input device")

        # Validate the entire mask BEFORE modifying cache state.
        mask = self._allowed_mask(
            batch=B, q_len=T, kv_len=kv_len,
            positions=cache_position, device=hidden_states.device,
            attention_mask=attention_mask,
        )

        if self.q_proj is not None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_norm(self.q_a_proj(hidden_states)))
        q = q.reshape(B, T, H, dc + dr).transpose(1, 2)
        q_nope, q_rot = q.split((dc, dr), dim=-1)

        compressed = self.kv_a_proj(hidden_states)
        latent, k_rot = compressed.split(
            (self.config.kv_lora_rank, dr), dim=-1
        )
        latent = self.kv_a_norm(latent).unsqueeze(1)  # (B, 1, T, r)
        k_rot = k_rot.unsqueeze(1)                    # (B, 1, T, dR)

        cos, sin = self.rotary_emb(q_rot, cache_position)
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)

        if use_cache:
            # This is the actual compressed cache; do not repeat K/V heads.
            latent, k_rot = past_key_values.update(
                latent, k_rot, layer_idx=self.layer_idx,
                cache_kwargs={"cache_position": cache_position},
            )

        if mode == "naive":
            output = self._naive(q_nope, q_rot, latent, k_rot, mask)
        else:
            output = self._absorbed(q_nope, q_rot, latent, k_rot, mask)

        output = output.transpose(1, 2).contiguous().reshape(
            B, T, H * self.config.v_head_dim
        )
        return self.o_proj(output)
