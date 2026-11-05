from latent_forge.configs.mla import MLAConfig
from latent_forge.models.decoder import build_transformer

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
    N=6,
    h=8,
    dropout=0.0,
    d_ff=1024,
    attention_type="mla",
    position_encoding="rope",
    mla_config=config,
    norm_type="rmsnorm",
    ffn_type="swiglu",
)

print(type(model.decoder.norm).__name__)
print(type(model.decoder.layers[0].feed_forward_block).__name__)