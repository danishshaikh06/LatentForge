import torch 
from torch import nn
import math

from collections.abc import Mapping
from typing import Any, Literal

from transformers.cache_utils import Cache, DynamicCache

from latent_forge.configs.attention import AttentionConfig
from latent_forge.models.attention.gqa import GroupedQueryAttention
from latent_forge.models.attention.mha import MultiHeadAttention
from latent_forge.models.attention.rope import RotaryEmbedding
from latent_forge.configs.mla import MLAConfig
from latent_forge.models.attention.mla import MultiHeadLatentAttention

class InputEmbedding(nn.Module):

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size= vocab_size
        self.d_model= d_model
        self.embedding=nn.Embedding(vocab_size, d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model) 
    

class PositionalEncoding(nn.Module):
    def __init__(self,d_model: int, seq_len: int, dropout: float ) -> None: # dropout is used to make model less overfit
        super().__init__()
        self.d_model= d_model
        self.seq_len= seq_len   
        self.dropout= nn.Dropout(dropout)

        # create a matrix of size(seq_len,d_model)
        pe=torch.zeros(seq_len, d_model) 

        # create a vector of shape (seq_len,1) for the sequence word to be positionally embedded
        position=torch.arange(0,seq_len, dtype=torch.float).unsqueeze(1) # torch.arange is used to create a vector of size seq_len 
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:, 0::2]=torch.sin(position*div_term) # even indices
        pe[:, 1::2]=torch.cos(position*div_term) # odd indices

        # for batch of sentences 1-> represent batch number
        pe=pe.unsqueeze(0) # (1,seq_len, d_model) unsqueeze is used to add a dimension at the 0th index
       
        # register buffer is used to register a tensor as a buffer. Buffer is not a parameter
        self.register_buffer('pe',pe)

    def forward(self, x, start_pos: int = 0):

        length = x.size(1)

        if start_pos < 0 or start_pos + length > self.pe.size(1):
            raise ValueError(
                "Input exceeds the configured sequence length"
            )

        x = x + self.pe[
            :, start_pos:start_pos + length, :
        ]

        return self.dropout(x) 

NormType = Literal["layernorm", "rmsnorm"]
FFNType = Literal["relu", "swiglu"]

def create_norm(
    d_model: int,
    norm_type: NormType,
    eps: float = 1e-6,
) -> nn.Module:
    """
    Construct a decoder normalization layer.

    LayerNorm:
        Preserves the existing custom implementation.

    RMSNorm:
        Uses PyTorch's native implementation.
    """
    if norm_type == "layernorm":
        return LayerNorm(
            d_model,
            esp=eps,
        )

    if norm_type == "rmsnorm":
        return nn.RMSNorm(
            d_model,
            eps=eps,
        )

    raise ValueError(
        f"Unsupported normalization: {norm_type}"
    ) 

class LayerNorm(nn.Module):
    def __init__(self, d_model: int, esp: float = 1e-6) -> None :
        super().__init__()
        self.esp =  esp
        self.gamma= nn.Parameter(torch.ones(d_model)) # gamma (Multiplicative)
        self.beta= nn.Parameter(torch.zeros(d_model)) # beta (Additive)

    def forward(self,x):
        mean= x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False) 
        return self.gamma * (x - mean)/(torch.sqrt(var + self.esp)) + self.beta # why * with mean and + with bias? because we want to scale and shift the normalized value


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff:int, dropout: float=0.1) -> None:
        super().__init__()
        self.linear_1=nn.Linear(d_model, d_ff) # d_ff is usually 4 times of d_model
        self.dropout= nn.Dropout(dropout)
        self.linear_2=nn.Linear(d_ff, d_model)

    def forward(self,x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x)))) # will it returns as [d_model, d_model]? Yes, because the first linear layer transforms from d_model to d_ff, and the second linear layer transforms back from d_ff to d_model. 


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU feed-forward network.

    Architecture:
        gate = SiLU(W_gate(x))
        up   = W_up(x)
        out  = W_down(gate * up)

    Input:
        (B, T, d_model)

    Output:
        (B, T, d_model)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if (
            isinstance(d_model, bool)
            or not isinstance(d_model, int)
            or d_model <= 0
            or isinstance(d_ff, bool)
            or not isinstance(d_ff, int)
            or d_ff <= 0
        ):
            raise ValueError(
                "d_model and d_ff must be positive integers"
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1)"
            )

        self.d_model = d_model
        self.d_ff = d_ff

        self.gate_proj = nn.Linear(
            d_model,
            d_ff,
            bias=False,
        )

        self.up_proj = nn.Linear(
            d_model,
            d_ff,
            bias=False,
        )

        self.down_proj = nn.Linear(
            d_ff,
            d_model,
            bias=False,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        gate = torch.nn.functional.silu(
            self.gate_proj(x)
        )

        up = self.up_proj(x)

        hidden_states = gate * up

        hidden_states = self.dropout(
            hidden_states
        )

        return self.down_proj(
            hidden_states
        )

class ResidualConnection(nn.Module): 
    def __init__(
        self,
        d_model: int,
        dropout: float,
        norm_type: NormType = "layernorm",
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.norm = create_norm(
            d_model=d_model,
            norm_type=norm_type,
        )

    def forward(self, x, sublayer):
       return x + self.dropout(sublayer(self.norm(x)))


class DecoderBlock(nn.Module):
    """
    Pre-norm decoder block.

    Supports legacy MHA and the new configurable GQA module.
    """
    def __init__(
        self,
        d_model: int,
        self_attention_block: nn.Module,
        feed_forward_block: nn.Module,
        dropout: float,
        norm_type: NormType = "layernorm",
    ) -> None:
        super().__init__()

        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block

        self.residual_connections = nn.ModuleList([
            ResidualConnection(
                d_model,
                dropout,
                norm_type=norm_type,
            )
            for _ in range(2)
        ])

    def forward(
        self,
        x: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        *,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[
            torch.Tensor, torch.Tensor
        ] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:

        def apply_attention(
            hidden_states: torch.Tensor,
        ) -> torch.Tensor:

            attention = self.self_attention_block

            if isinstance(attention, GroupedQueryAttention):
                return attention(
                    hidden_states,
                    attention_mask=tgt_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    use_cache=use_cache,
                )

            if isinstance(attention, MultiHeadLatentAttention):
                # MLA applies its own decoupled RoPE internally.
                if position_embeddings is not None:
                    raise ValueError(
                        "MLA must not receive external RoPE embeddings"
                    )

                return attention(
                    hidden_states,
                    attention_mask=tgt_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    use_cache=use_cache,
                )

            # Legacy MHA.
            if use_cache:
                raise NotImplementedError(
                    "Legacy MHA does not support KV caching"
                )

            output, _ = attention(
                hidden_states,
                hidden_states,
                hidden_states,
                tgt_mask,
            )
            return output

        x = self.residual_connections[0](
            x,
            apply_attention,
        )

        x = self.residual_connections[1](
            x,
            self.feed_forward_block,
        )

        return x


class Decoder(nn.Module):
    """
    Stack of Transformer decoder layers.

    One request-owned cache object is shared across
    layers, but every layer updates a separate cache entry.
    """
    def __init__(
        self,
        layers: nn.ModuleList,
        features: int,
        norm_type: NormType = "layernorm",
    ) -> None:
        super().__init__()

        self.layers = layers

        self.norm = create_norm(
            d_model=features,
            norm_type=norm_type,
        )

    def forward(
        self,
        x: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        *,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:

        for layer in self.layers:
            x = layer(
                x,
                tgt_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                use_cache=use_cache,
            )

        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()  
        self.proj = nn.Linear(d_model, vocab_size)
        

    def forward(self, x):
        logits = self.proj(x) 
        
        return logits


class Transformer(nn.Module):

    def __init__(
        self,
        decoder: Decoder,
        tgt_embed: InputEmbedding,
        tgt_pos: PositionalEncoding | None,
        projection_layer: ProjectionLayer,
        rotary_emb: RotaryEmbedding | None = None,
        *,
        attention_type: Literal["mha", "gqa", "mla"] = "gqa",
    ) -> None:
        super().__init__()

        if attention_type not in ("mha", "gqa", "mla"):
            raise ValueError("Invalid attention type")

        if attention_type == "mla":
            # MLA owns its decoupled RoPE.
            if tgt_pos is not None or rotary_emb is not None:
                raise ValueError(
                    "MLA must not use external positional encoding"
                )
        else:
            # Preserve existing positional-encoding behavior.
            if (tgt_pos is None) == (rotary_emb is None):
                raise ValueError(
                    "Exactly one positional encoding method is required"
                )

        self.attention_type = attention_type
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.tgt_pos = tgt_pos
        self.rotary_emb = rotary_emb
        self.projection_layer = projection_layer
    
    
    def decode(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        *,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:

        past_len = (
            0
            if past_key_values is None
            else past_key_values.get_seq_length(0)
        )

        hidden_states = self.tgt_embed(tgt)

        if cache_position is None:
            cache_position = torch.arange(
                past_len,
                past_len + tgt.shape[1],
                dtype=torch.long,
                device=tgt.device,
            )

        if self.attention_type == "mla":
            # MLA computes its own RoPE using cache_position.
            # Never add sinusoidal PE or external RoPE here.
            position_embeddings = None

        elif self.rotary_emb is not None:
            cos, sin = self.rotary_emb(
                hidden_states,
                position_ids=cache_position,
            )
            position_embeddings = (cos, sin)

        else:
            if self.tgt_pos is None:
                raise RuntimeError(
                    "Positional encoding was not initialized"
                )

            hidden_states = self.tgt_pos(
                hidden_states,
                start_pos=past_len,
            )
            position_embeddings = None

        return self.decoder(
            hidden_states,
            tgt_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
        )


    
    @staticmethod
    def create_tgt_mask(tgt, pad_token_id):
        # tgt: (B, tgt_seq_len)
        pad_mask = (tgt != pad_token_id).unsqueeze(1).unsqueeze(2)  

        seq_len = tgt.size(1)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=tgt.device)).bool()  # (tgt_seq_len, tgt_seq_len)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1) 

        return pad_mask & causal_mask  # (B, 1, tgt_seq_len, tgt_seq_len)

    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    
    def forward(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        *,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Cache]:
        """
            Full-sequence training or cached autoregressive inference.

            Without cache:
                Returns logits (B, T, vocab_size).

            With cache:
                Returns (logits, updated_cache).
        """

        if tgt.ndim != 2 or tgt.dtype != torch.long:
            raise ValueError(
                "tgt must be a LongTensor of shape (B, T)"
            )

        if tgt.shape[0] == 0 or tgt.shape[1] == 0:
            raise ValueError("Empty token batches are unsupported")

        if not use_cache and past_key_values is not None:
            raise ValueError(
                "Cannot supply past_key_values "
                "when use_cache=False"
            )

        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()

            if not isinstance(past_key_values, DynamicCache):
                raise NotImplementedError(
                    "Only DynamicCache is supported currently"
                )

            # Catch partially updated or inconsistent caches.
            start_pos = past_key_values.get_seq_length(0)

            for layer_idx in range(len(self.decoder.layers)):
                if (
                    past_key_values.get_seq_length(layer_idx)
                    != start_pos
                ):
                    raise RuntimeError(
                        "Inconsistent KV-cache sequence lengths "
                        "across decoder layers"
                    )

            cache_position = torch.arange(
                start_pos,
                start_pos + tgt.shape[1],
                device=tgt.device,
                dtype=torch.long,
            )

        else:
            cache_position = None

        hidden_states = self.decode(
            tgt,
            tgt_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=use_cache,
        )

        logits = self.project(hidden_states)

        if use_cache:
            return logits, past_key_values

        return logits

def build_transformer(
    tgt_vocab_size: int,
    tgt_seq_len: int,
    d_model: int,
    N: int,
    h: int,
    dropout: float,
    d_ff: int = 2048,
    kv_heads: int | None = None,
    attention_backend: str = "sdpa",
    position_encoding: str = "sinusoidal",
    rope_theta: float = 10000.0,
    rotary_dim: int | None = None,
    attention_type: Literal["mha", "gqa", "mla"] | None = None,
    mla_config: MLAConfig | None = None,
    norm_type: NormType = "layernorm",
    ffn_type: FFNType = "relu",
) -> Transformer:
    if norm_type not in ("layernorm", "rmsnorm"):
        raise ValueError(
            "norm_type must be 'layernorm' or 'rmsnorm'"
        )

    if ffn_type not in ("relu", "swiglu"):
        raise ValueError(
            "ffn_type must be 'relu' or 'swiglu'"
        )
    if N <= 0:
        raise ValueError("N must be positive")

    if position_encoding not in ("sinusoidal", "rope"):
        raise ValueError("Invalid positional encoding")

    # Preserve old builder behavior.
    if attention_type is None:
        attention_type = (
            "gqa" if kv_heads is not None else "mha"
        )

    if attention_type not in ("mha", "gqa", "mla"):
        raise ValueError("Invalid attention_type")

    if attention_type != "mla" and mla_config is not None:
        raise ValueError(
            "mla_config is only valid for MLA models"
        )

    attention_config = None

    if attention_type == "mha":
        if kv_heads is not None:
            raise ValueError(
                "kv_heads must be None for legacy MHA"
            )
        if position_encoding != "sinusoidal":
            raise ValueError(
                "Legacy MHA supports sinusoidal PE only"
            )

    elif attention_type == "gqa":
        if kv_heads is None:
            raise ValueError(
                "GQA requires kv_heads"
            )

        attention_config = AttentionConfig(
            d_model=d_model,
            num_attention_heads=h,
            num_key_value_heads=kv_heads,
            dropout=dropout,
            backend=attention_backend,
        )

    elif attention_type == "mla":
        if mla_config is None:
            raise ValueError(
                "MLA requires mla_config"
            )

        if kv_heads is not None:
            raise ValueError(
                "kv_heads must be None for MLA"
            )

        if (
            mla_config.hidden_size != d_model
            or mla_config.num_attention_heads != h
        ):
            raise ValueError(
                "MLA configuration must match "
                "d_model and attention heads"
            )

        if position_encoding != "rope":
            raise ValueError(
                "MLA requires position_encoding='rope'"
            )

        if rotary_dim is not None:
            raise ValueError(
                "Use MLAConfig.qk_rope_head_dim "
                "instead of rotary_dim"
            )
        
    tgt_embed = InputEmbedding(
        tgt_vocab_size,
        d_model,
    )

    if attention_type == "mla":
        # MLA has its own decoupled RoPE per layer.
        tgt_pos = None
        rotary_emb = None

    elif position_encoding == "rope":
        if rotary_dim is not None and rotary_dim % 2:
            raise ValueError("rotary_dim must be even")

        tgt_pos = None
        rotary_emb = RotaryEmbedding(
            head_dim=attention_config.head_dim,
            theta=rope_theta,
            rotary_dim=rotary_dim,
        )

    else:
        if rotary_dim is not None:
            raise ValueError(
                "rotary_dim is only valid with RoPE"
            )

        tgt_pos = PositionalEncoding(
            d_model,
            tgt_seq_len,
            dropout,
        )
        rotary_emb = None

    #Decoder layers 
    decoder_blocks = []

    for layer_idx in range(N):

        if attention_type == "mla":
            attention = MultiHeadLatentAttention(
                mla_config,
                layer_idx=layer_idx,
            )

        elif attention_type == "gqa":
            attention = GroupedQueryAttention(
                attention_config,
                layer_idx=layer_idx,
            )

        else:
            attention = MultiHeadAttention(
                d_model,
                h,
                dropout,
            )

        if ffn_type == "swiglu":
            ffn = SwiGLUFeedForward(
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
            )
        else:
            ffn = FeedForward(
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
            )

        decoder_blocks.append(
            DecoderBlock(
                d_model=d_model,
                self_attention_block=attention,
                feed_forward_block=ffn,
                dropout=dropout,
                norm_type=norm_type,
            )
        )

    decoder = Decoder(
        layers=nn.ModuleList(decoder_blocks),
        features=d_model,
        norm_type=norm_type,
    )

    projection_layer = ProjectionLayer(
        d_model,
        tgt_vocab_size,
    )

    transformer = Transformer(
        decoder=decoder,
        tgt_embed=tgt_embed,
        tgt_pos=tgt_pos,
        projection_layer=projection_layer,
        rotary_emb=rotary_emb,
        attention_type=attention_type,
    )

    # Preserve tied embedding/LM-head weights.
    transformer.projection_layer.proj.weight = (
        transformer.tgt_embed.embedding.weight
    )

    for parameter in transformer.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)

    return transformer

def build_transformer_from_config(
    config: Mapping[str, Any],
    *,
    tgt_vocab_size: int,
) -> Transformer:
    """
    Reconstruct a model from training metadata.

    Supports older configs that predate attention_type.
    """
    attention_type = config.get("attention_type")

    if attention_type is None:
        attention_type = (
            "gqa"
            if config.get("kv_heads") is not None
            else "mha"
        )

    mla_config = None

    if attention_type == "mla":
        raw = config.get("mla")

        if not isinstance(raw, Mapping):
            raise ValueError(
                "MLA checkpoint requires an 'mla' config object"
            )

        mla_config = MLAConfig(
            hidden_size=config["d_model"],
            num_attention_heads=config["heads"],
            kv_lora_rank=raw["kv_lora_rank"],
            qk_nope_head_dim=raw["qk_nope_head_dim"],
            qk_rope_head_dim=raw["qk_rope_head_dim"],
            v_head_dim=raw["v_head_dim"],
            q_lora_rank=raw.get("q_lora_rank"),
            rope_theta=raw.get(
                "rope_theta",
                config.get("rope_theta", 10000.0),
            ),
            attention_dropout=raw.get(
                "attention_dropout", 0.0
            ),
            implementation=raw.get(
                "implementation", "naive"
            ),
        )

    return build_transformer(
        tgt_vocab_size=tgt_vocab_size,
        tgt_seq_len=config["seq_len"],
        d_model=config["d_model"],
        N=config["layers"],
        h=config["heads"],
        dropout=config["dropout"],
        d_ff=config["d_ff"],
        kv_heads=config.get("kv_heads"),
        attention_backend=config.get(
            "attention_backend", "sdpa"
        ),
        position_encoding=config.get(
            "position_encoding", "sinusoidal"
        ),
        rope_theta=config.get("rope_theta", 10000.0),
        rotary_dim=config.get("rotary_dim"),
        attention_type=attention_type,
        mla_config=mla_config,
        norm_type=config.get(
            "norm_type", "layernorm"
        ),
        ffn_type=config.get(
            "ffn_type", "relu"
        ),
    )