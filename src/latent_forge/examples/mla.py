import torch
from transformers.cache_utils import DynamicCache

from latent_forge.configs.mla import MLAConfig
from latent_forge.models.decoder import build_transformer


device = torch.device("cuda")

config = MLAConfig(
    hidden_size=384,
    num_attention_heads=8,
    q_lora_rank=96,
    kv_lora_rank=96,
    qk_nope_head_dim=48,
    qk_rope_head_dim=16,
    v_head_dim=48,
    implementation="absorbed",
)

model = build_transformer(
    tgt_vocab_size=4096,
    tgt_seq_len=256,
    d_model=384,
    N=4,
    h=8,
    dropout=0.0,
    d_ff=1536,
    attention_type="mla",
    position_encoding="rope",
    mla_config=config,
).to(device).eval()

cache = DynamicCache()

with torch.inference_mode():
    # Prefill
    prompt = torch.randint(
        0, 4096, (2, 64), device=device
    )

    logits, cache = model(
        prompt,
        past_key_values=cache,
        use_cache=True,
    )

    print("Prefill:", logits.shape)

    # Decode
    token = torch.randint(
        0, 4096, (2, 1), device=device
    )

    logits, cache = model(
        token,
        past_key_values=cache,
        use_cache=True,
    )

    print("Decode:", logits.shape)

    for layer_idx in range(4):
        latent, rotary_key = cache[layer_idx]

        print(
            f"Layer {layer_idx}",
            latent.shape,
            rotary_key.shape,
        )