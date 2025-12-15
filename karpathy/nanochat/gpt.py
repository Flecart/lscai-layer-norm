"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Multi-Query Attention (MQA) support for more efficient inference
"""

import math
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW

from nanochat.owned.normalization_strategy import LearnableRMSNormStrategy, NormType, build_norm_strategy
from nanochat.owned.mlp_fused_strategy import BlockMLPStrategy, ColumnMLPFusedStrategy, FullMLPFusedStrategy, RowMLPFusedStrategy, build_mlp_strategy

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (MQA)
    n_embd: int = 768

    norm_type: str = "rms"  # "rms", "layernorm", "none"
    norm_eps: float | None = None  # used e.g. for RMSNorm
    
    mlp_type: str = "default"  # "default", "column", "row", "full", "patched"
    mlp_type_qkv: str = "default"  # "default", "column", "row", "full", "patched"

    embed_norm_type: str = "rms"
    final_norm_type: str = "rms"

    group_size_torus_norm: int = 2    # only used if norm_type is "torus"
    attention_strategy: str = "default"  # "default", "pair_scaled"
    
    # ANGELO: rms was the Karpathy default one, I don't think we need to mess with this norm.
    qk_norm_type: str | None = "rms"  # if None, reuse norm_type
    pre_attn_norm_type: str = "rms" # I would also keep this stable, so that our experiment is only changing the mlp norm.
    

    use_muon: str = "false"
    
    init_type: str = "scaled"  # "scaled", "xavier", "kaiming"



def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3) # re-assemble
    out = out.to(x.dtype) # ensure input/output dtypes match
    return out



class PairScaledCausalSelfAttention(nn.Module):
    """
    Causal self-attention with:
    - rotary embeddings
    - optional QK normalization strategy (RMS/LAYER/TORUS/NONE)
    - learnable per-2D-pair scaling in the attention score:
        score = sum_i alpha_i * (q[2i:2i+2] · k[2i:2i+2])

    Efficient implementation:
    - learn log_alpha over pairs
    - scale q and k by exp(0.5 * log_alpha) per pair (broadcasted)
    - then call F.scaled_dot_product_attention (still fast)
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head

        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        assert self.head_dim % 2 == 0, "Pair scaling assumes head_dim is even (pairs of 2)."

        if config.mlp_type != "default":
            assert self.n_head == self.n_kv_head, (
                "When using fused MLP strategies, please set n_head == n_kv_head for simplicity."
            )

        # Projections
        self.c_q = build_mlp_strategy(config.mlp_type_qkv, self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = build_mlp_strategy(config.mlp_type_qkv, self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = build_mlp_strategy(config.mlp_type_qkv, self.n_embd, self.n_kv_head * self.head_dim, bias=False)

        # tie weights for q,k,v projections if column_parameter is present
        if hasattr(self.c_q, "column_parameters") and hasattr(self.c_k, "column_parameters") and hasattr(self.c_v, "column_parameters"):
            self.c_k.column_parameters = self.c_q.column_parameters
            self.c_v.column_parameters = self.c_q.column_parameters

        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        # QK norm
        self.qk_norm = build_norm_strategy(
            NormType(config.qk_norm_type),
            self.head_dim,
            eps=config.norm_eps,
            group_size_torus_norm=config.group_size_torus_norm,
        )

        # Learnable per-pair weights (shared across heads; simplest + fast)
        n_pairs = self.head_dim // 2
        self.log_alpha = nn.Parameter(torch.zeros(n_pairs))  # alpha = exp(log_alpha), init alpha=1

    def _apply_pair_scaling(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (..., head_dim)
        scales pairs (0,1), (2,3), ... by sqrt(alpha_i)
        """
        # scale_per_pair = sqrt(alpha) = exp(0.5 * log_alpha)
        scale_per_pair = torch.exp(0.5 * self.log_alpha)  # (n_pairs,)
        scale = scale_per_pair.repeat_interleave(2)        # (head_dim,)
        return t * scale

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        # Project
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Rotary
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # QK norm
        q = self.qk_norm(q)
        k = self.qk_norm(k)

        # Pair scaling (broadcast across batch/time/heads)
        q = self._apply_pair_scaling(q)
        k = self._apply_pair_scaling(k)

        # (B, T, H, D) -> (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # KV cache
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)

        Tq = q.size(2)
        Tk = k.size(2)

        enable_gqa = self.n_head != self.n_kv_head
        if kv_cache is None or Tq == Tk:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
            prefix_len = Tk - Tq
            if prefix_len > 0:
                attn_mask[:, :prefix_len] = True
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # Re-assemble + proj
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        
        if config.mlp_type != "default":
            assert self.n_head == self.n_kv_head, "When using fused MLP strategies, please set n_head == n_kv_head for simplicity."
        
        # self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_q = build_mlp_strategy(
            config.mlp_type_qkv,
            self.n_embd,
            self.n_head * self.head_dim,
            bias=False
        )
        # self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_k = build_mlp_strategy(
            config.mlp_type_qkv,
            self.n_embd,
            self.n_kv_head * self.head_dim,
            bias=False
        )
        # self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = build_mlp_strategy(
            config.mlp_type_qkv,
            self.n_embd,
            self.n_kv_head * self.head_dim,
            bias=False
        )
        
        # tie weights for q,k,v projections if column_parameter is present
        # this should make it identical to using nn.Linear and lrms
        if hasattr(self.c_q, "column_parameters") and hasattr(self.c_k, "column_parameters") and hasattr(self.c_v, "column_parameters"):
            self.c_k.column_parameters = self.c_q.column_parameters
            self.c_v.column_parameters = self.c_q.column_parameters
        
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        self.qk_norm = build_norm_strategy(
            NormType(config.qk_norm_type),
            self.head_dim,
            eps=config.norm_eps,
            group_size_torus_norm=config.group_size_torus_norm,
        )


    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding

        # QK norm (strategy can be identity to disable)
        q = self.qk_norm(q)
        k = self.qk_norm(k)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2) # number of queries in this forward pass
        Tk = k.size(2) # number of keys/values in total (in the cache + current forward pass)

        # Attention: queries attend to keys/values autoregressively. A few cases to handle:
        enable_gqa = self.n_head != self.n_kv_head # Group Query Attention (GQA): duplicate key/value heads to match query heads if desired
        if kv_cache is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            # And even if there is KV cache, we can still use this simple version when Tq == Tk
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            # During inference AND we have a chunk of queries in this forward pass:
            # First, each query attends to all the cached keys/values (i.e. full prefix)
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device) # True = keep, False = mask
            prefix_len = Tk - Tq
            if prefix_len > 0: # can't be negative but could be zero
                attn_mask[:, :prefix_len] = True
            # Then, causal attention within this chunk
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = build_mlp_strategy(
            config.mlp_type,
            config.n_embd,
            4 * config.n_embd,
            bias=False
        )
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()

        if config.attention_strategy == "pair_scaled":
            self.attn = PairScaledCausalSelfAttention(config, layer_idx)
        elif config.attention_strategy == "default":
            self.attn = CausalSelfAttention(config, layer_idx)
        else:
            raise ValueError(f"Unknown attention_strategy: {config.attention_strategy}")

        self.mlp = MLP(config)

        # Removing this for simplicity of the analysis.
        # pre-attn and pre-mlp norms (and you can add post norms if you want later)
        self.pre_attn_norm = build_norm_strategy(
            NormType(config.pre_attn_norm_type),
            config.n_embd,
            eps=config.norm_eps,
            group_size_torus_norm=config.group_size_torus_norm,
        )
        self.pre_mlp_norm = build_norm_strategy(
            NormType(config.norm_type),
            config.n_embd,
            eps=config.norm_eps,
            group_size_torus_norm=config.group_size_torus_norm,
        )


    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(self.pre_attn_norm(x), cos_sin, kv_cache)
        x = x + self.mlp(self.pre_mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # To support meta device initialization, we init the rotary embeddings here, but it's fake
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)


        # ANGELO: DELETING THESE, THESE ARE NOT FOCUS OF OUR ANALYSIS
        # self.embed_norm = build_norm_strategy(
        #     NormType(config.norm_type),
        #     config.n_embd,
        #     eps=config.norm_eps,
        # )

        # self.final_norm = build_norm_strategy(
        #     NormType(config.norm_type),
        #     config.n_embd,
        #     eps=config.norm_eps,
        # )
        self.embed_norm = build_norm_strategy(
            NormType(config.embed_norm_type),
            config.n_embd,
            eps=config.norm_eps,
            group_size_torus_norm=config.group_size_torus_norm,
        )
        self.final_norm = build_norm_strategy(
            NormType(config.final_norm_type),
            config.n_embd,
            eps=config.norm_eps,
            group_size_torus_norm=config.group_size_torus_norm,
        )

        # Debug flag to avoid spamming prints every forward
        self._printed_embed_norm_debug = False


    def init_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        torch.nn.init.zeros_(self.lm_head.weight)
        # zero out c_proj weights in all blocks
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory: both in the model and the activations
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

        # --- DEBUG: verify scale params are all ones ---
        ddp, rank, *_ = get_dist_info()
        is_master = (not ddp) or (rank == 0)

        if is_master:
            with torch.no_grad():
                for name, module in self.named_modules():
                    if isinstance(module, LearnableRMSNormStrategy):
                        g = module.gamma
                        if not torch.allclose(g, torch.ones_like(g), atol=1e-6):
                            raise RuntimeError(
                                f"[RMSNorm gamma check] {name}.gamma is not all ones. "
                                f"mean={g.mean().item():.6f}, min={g.min().item():.6f}, max={g.max().item():.6f}"
                            )
                        else:
                            print(f"[RMSNorm gamma check] {name}.gamma OK (all ones, dim={g.numel()})")
                    elif isinstance(module, ColumnMLPFusedStrategy):
                        p = module.norm_parameters
                        if not torch.allclose(p, torch.ones_like(p), atol=1e-6):
                            raise RuntimeError(
                                f"[ColumnMLPFusedStrategy norm_parameters check] {name}.norm_parameters is not all ones. "
                                f"mean={p.mean().item():.6f}, min={p.min().item():.6f}, max={p.max().item():.6f}"
                            )
                        else:
                            print(f"[ColumnMLPFusedStrategy norm_parameters check] {name}.norm_parameters OK (all ones, dim={p.numel()})")
                    elif isinstance(module, RowMLPFusedStrategy):
                        p = module.norm_parameters
                        if not torch.allclose(p, torch.ones_like(p), atol=1e-6):
                            raise RuntimeError(
                                f"[RowMLPFusedStrategy norm_parameters check] {name}.norm_parameters is not all ones. "
                                f"mean={p.mean().item():.6f}, min={p.min().item():.6f}, max={p.max().item():.6f}"
                            )
                        else:
                            print(f"[RowMLPFusedStrategy norm_parameters check] {name}.norm_parameters OK (all ones, dim={p.numel()})")
                    elif isinstance(module, FullMLPFusedStrategy):
                        cp = module.column_parameters
                        rp = module.row_parameters
                        if not torch.allclose(cp, torch.ones_like(cp), atol=1e-6):
                            raise RuntimeError(
                                f"[FullMLPFusedStrategy column_parameters check] {name}.column_parameters is not all ones. "
                                f"mean={cp.mean().item():.6f}, min={cp.min().item():.6f}, max={cp.max().item():.6f}"
                            )
                        else:
                            print(f"[FullMLPFusedStrategy column_parameters check] {name}.column_parameters OK (all ones, dim={cp.numel()})")
                        if not torch.allclose(rp, torch.ones_like(rp), atol=1e-6):
                            raise RuntimeError(
                                f"[FullMLPFusedStrategy row_parameters check] {name}.row_parameters is not all ones. "
                                f"mean={rp.mean().item():.6f}, min={rp.min().item():.6f}, max={rp.max().item():.6f}"
                            )
                        else:
                            print(f"[FullMLPFusedStrategy row_parameters check] {name}.row_parameters OK (all ones, dim={rp.numel()})")
                    elif isinstance(module, BlockMLPStrategy):
                        bs = module.block_scale
                        if not torch.allclose(bs, torch.ones_like(bs), atol=1e-6):
                            raise RuntimeError(
                                f"[BlockMLPStrategy block_scale check] {name}.block_scale is not all ones. "
                                f"mean={bs.mean().item():.6f}, min={bs.min().item():.6f}, max={bs.max().item():.6f}"
                            )
                        else:
                            print(f"[BlockMLPStrategy block_scale check] {name}.block_scale OK (all ones, shape={bs.shape})")
                    elif isinstance(module, PairScaledCausalSelfAttention):
                        la = module.log_alpha
                        if not torch.allclose(la, torch.zeros_like(la), atol=1e-6):
                            raise RuntimeError(
                                f"[PairScaledAttn log_alpha check] {name}.log_alpha is not all zeros. "
                                f"mean={la.mean().item():.6f}, min={la.min().item():.6f}, max={la.max().item():.6f}"
                            )
                        else:
                            print(f"[PairScaledAttn log_alpha check] {name}.log_alpha OK (all zeros, n={la.numel()})")

        # --- END DEBUG ---


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)

            itype = getattr(self.config, "init_type", "scaled")

            if itype == "scaled":
                # your current default
                std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            elif itype == "xavier":
                # Xavier / Glorot normal
                std = math.sqrt(2.0 / (fan_in + fan_out))
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            elif itype == "kaiming":
                # Kaiming He normal (for ReLU-style activations)
                std = math.sqrt(2.0 / fan_in)
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            else:
                raise ValueError(f"Unknown init_type: {itype}")

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            # keep this constant across inits to isolate the effect on linear layers
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

        elif isinstance(module, LearnableRMSNormStrategy):
            nn.init.ones_(module.gamma)

        elif isinstance(module, ColumnMLPFusedStrategy):
            nn.init.ones_(module.norm_parameters)
            self._init_weights(module.linear)

        elif isinstance(module, RowMLPFusedStrategy):
            nn.init.ones_(module.norm_parameters)
            self._init_weights(module.linear)
            
        elif isinstance(module, FullMLPFusedStrategy):
            nn.init.ones_(module.column_parameters)
            nn.init.ones_(module.row_parameters)
            self._init_weights(module.linear)

        elif isinstance(module, BlockMLPStrategy):
            nn.init.ones_(module.block_scale)
            self._init_weights(module.linear)
        
        elif isinstance(module, PairScaledCausalSelfAttention):
            nn.init.zeros_(module.log_alpha)  # alpha = exp(0) = 1



    # TODO: bump base theta more, e.g. 100K is more common more recently
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """ Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311 """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into 3 groups (matrix, embedding, lm_head)
        block_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        if rank == 0:
            # I was debugging the extra embed and final norm here....
            print(
                f"Rank {rank}: Optimizer parameter groups: \n"
                f"block_params={len(block_params)}, \n"
                f"embedding_params={len(embedding_params)}, \n"
                f"lm_head_params={len(lm_head_params)}\n"
                f"full model_params={len(list(self.parameters()))}"
            )
        assert len(list(self.parameters())) == len(block_params) + len(embedding_params) + len(lm_head_params)
        # Create the AdamW optimizer for the embedding and lm_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Create the Muon optimizer for the linear layers
        block_params = (
            list(self.transformer.h.parameters())
            + list(self.embed_norm.parameters())
            + list(self.final_norm.parameters())
        )
        matrix_params = [p for p in block_params if p.ndim == 2]
        rmsnorm_params = [p for p in block_params if p.ndim == 1]
        if rank == 0:
            print(f"Total block params: {len(block_params)}")
            print(f"Muon? optimizer will optimize {len(matrix_params)} matrix params")
            print(f"AdamW optimizer will optimize {len(rmsnorm_params)} RMSNorm params")

            # Print out which optimizer is assigned to each parameter in block 0 for verification
            for name, param in self.transformer.h[0].named_parameters():
                if param.ndim == 2 and self.config.use_muon == "true":
                    opt_name = "Muon"
                else:
                    opt_name = "AdamW"
                print(f"Block 0 param: {name}, shape={param.shape}, optimizer={opt_name}")

        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            dict(params=rmsnorm_params, lr=matrix_lr * dmodel_lr_scale),
        ]
        optimizers = []
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        if self.config.use_muon == "true":
            
            if rank == 0:
                print(f"Muon optimizer will optimize {len(matrix_params)} matrix params")
                print(f"AdamW optimizer will optimize {len(rmsnorm_params)} RMSNorm params")
            muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
            optimizers.append(muon_optimizer)
        else:
            # see muon init, copying that part here.
            for size in {p.numel() for p in matrix_params}:
                group = dict(params=[p for p in matrix_params if p.numel() == size], lr=matrix_lr * dmodel_lr_scale)
                adam_groups.append(group)

        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        optimizers.append(adamw_optimizer)
        
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)

        # --- DEBUG: check embed_norm behavior once on rank 0 ---
        if not self._printed_embed_norm_debug:
            ddp, rank, *_ = get_dist_info()
            if (not ddp) or (rank == 0):
                print("DEBUG: wte norm before embed_norm:", x.norm().item())
            x = self.embed_norm(x)
            if (not ddp) or (rank == 0):
                print(
                    "DEBUG: after embed_norm norm:",
                    x.norm().item(),
                    "max abs:",
                    x.abs().max().item(),
                )
            self._printed_embed_norm_debug = True
        else:
            x = self.embed_norm(x)
        # --- END DEBUG ---

        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = self.final_norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15
        if targets is not None:
            # training mode: compute and return the loss
            # TODO: experiment with Liger Kernels / chunked cross-entropy etc.
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap) # logits softcap
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference mode: compute and return the logits
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap) # logits softcap
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
