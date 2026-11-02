
import torch
from transformers.cache_utils import DynamicCache

from latent_forge.configs.attention import AttentionConfig
from latent_forge.models.attention.gqa import GroupedQueryAttention

device = "cuda" if torch.cuda.is_available() else "cpu"

config = AttentionConfig(
    d_model=512,
    num_attention_heads=8,
    num_key_value_heads=2,
    backend="sdpa",
)

attention = GroupedQueryAttention(
    config,
    layer_idx=0,
).to(device).eval()

cache = DynamicCache()

# Prefill
hidden = torch.randn(1, 10, 512, device=device)

with torch.inference_mode():
    output = attention(
        hidden,
        past_key_values=cache,
        use_cache=True,
    )

    print("Prefill output:", output.shape)
    print("Cached tokens:", cache.get_seq_length(0))

    # Decode a newly supplied hidden state
    next_hidden = torch.randn(1, 1, 512, device=device)

    output = attention(
        next_hidden,
        past_key_values=cache,
        cache_position=torch.tensor(
            [10], device=device, dtype=torch.long
        ),
        use_cache=True,
    )

    print("Decode output:", output.shape)
    print("Cached tokens:", cache.get_seq_length(0))
    print("KV shape:", cache[0][0].shape)
