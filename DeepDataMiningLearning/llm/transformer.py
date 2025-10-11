# ============================================================
# 1) MODEL BUILDER — Transformer LM (decoder-only, GPT-style)
# ============================================================
from __future__ import annotations
from tqdm.auto import tqdm
import math, os, json, time, random
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------- Small utils ----------
def exists(x): return x is not None
def default(x, d): return x if exists(x) else d

# ---------- Norms ----------
class RMSNorm(nn.Module):
    """RMSNorm from GPT-NeoX / Llama (no bias)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * (x * norm)

# ---------- SwiGLU FFN ----------
class SwiGLU(nn.Module):
    """SwiGLU MLP used in PaLM/Llama2."""
    def __init__(self, dim, hidden_mult=4, dropout=0.0):
        super().__init__()
        hidden = int(hidden_mult * dim * 2/3)  # typical compression
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = F.silu(self.w1(x)) * self.w2(x)
        return self.dropout(self.w3(x))

# ---------- (Optional) Rotary Embeddings (RoPE) ----------
class RotaryEmbedding(nn.Module):
    """Minimal RoPE for Q,K (GPT-NeoX/Llama style)."""
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [T, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)           # [T, dim]
        cos = emb.cos()[:, None, None, :]                 # [T,1,1,dim]
        sin = emb.sin()[:, None, None, :]
        return cos, sin

# def apply_rope(x, cos, sin):
#     # x: [B, nH, T, Hd]; split last dim into even/odd
#     x1, x2 = x[..., ::2], x[..., 1::2]
#     x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)
#     return (x * cos) + (x_rot * sin)

def apply_rope(x, cos, sin):
    """
    x: [B, nH, T, Hd]
    cos/sin: [T, 1, 1, Hd]
    """
    # Align cos/sin to sequence dim (2)
    cos = cos.permute(1, 2, 0, 3)  # -> [1, 1, T, Hd]
    sin = sin.permute(1, 2, 0, 3)  # -> [1, 1, T, Hd]

    # Split even/odd dims
    x1, x2 = x[..., ::2], x[..., 1::2]
    x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)
    return (x * cos) + (x_rot * sin)

# ---------- Multi-Head Attention (Flash when available) ----------
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.0, rope=False):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim) if rope else None

    def forward(self, x, attn_mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # [B,T,C] each
        # reshape to heads
        def split(t): return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,T,Hd]
        q, k, v = split(q), split(k), split(v)

        if self.rope is not None:
            cos, sin = self.rope(T, x.device)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        # Try PyTorch 2.0 SDPA (FlashAttention when available on device)
        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p, is_causal=attn_mask is None)
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            att = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B,H,T,T]
            if attn_mask is not None: att += attn_mask
            att = att.softmax(dim=-1)
            att = self.dropout(att)
            out = torch.matmul(att, v)  # [B,H,T,Hd]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

# ---------- Transformer Block (PreNorm, RMSNorm + SwiGLU) ----------
class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_mult=4.0, dropout=0.0, rope=False):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads, dropout=dropout, rope=rope)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, hidden_mult=mlp_mult, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, attn_mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), attn_mask))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

# ---------- Language Model (decoder-only) ----------
@dataclass
class ModelConfig:
    vocab_size: int
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    mlp_mult: float = 4.0
    dropout: float = 0.1
    max_seq_len: int = 1024
    rope: bool = True
    tie_weights: bool = True

class TransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.dim) if not cfg.rope else None
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.dim, cfg.n_heads, cfg.mlp_mult, cfg.dropout, rope=cfg.rope)
            for _ in range(cfg.n_layers)
        ])
        self.norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, attn_mask=None):
        B, T = x.shape
        assert T <= self.cfg.max_seq_len
        h = self.token_emb(x)
        if self.pos_emb is not None:
            pos = torch.arange(T, device=x.device)
            h = h + self.pos_emb(pos)[None, :, :]
        for blk in self.blocks:
            h = blk(h, attn_mask=attn_mask)
        h = self.norm(h)
        return self.lm_head(h)  # [B,T,V]

    @staticmethod
    def causal_mask(T, device):
        # upper-triangular True above diagonal -> convert to -inf mask
        mask = torch.full((T, T), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

# ============================================================
# Traditional Transformer (baseline GPT-style, 2017 architecture)
# ============================================================
class TraditionalTransformerLM(nn.Module):
    """
    Traditional decoder-only Transformer for language modeling.

    Key differences vs TransformerLM:
      - Uses LayerNorm instead of RMSNorm
      - Uses standard GELU MLP instead of SwiGLU
      - Uses absolute sinusoidal positional embeddings (no RoPE)
      - Architecture similar to GPT-2 / "Attention is All You Need" baseline
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        # --- Embedding layers ---
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.drop = nn.Dropout(dropout)

        # --- Transformer layers ---
        self.blocks = nn.ModuleList(
            [
                TransformerBlockClassic(dim, n_heads, ff_dim, dropout)
                for _ in range(n_layers)
            ]
        )

        # --- Final LayerNorm and output projection ---
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # weight tying

        # Initialize parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @staticmethod
    def causal_mask(seq_len, device):
        # Same as in TransformerLM
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: [B, T] token indices
            attn_mask: optional causal mask
        Returns:
            logits: [B, T, V]
        """
        B, T = x.shape
        device = x.device
        pos = torch.arange(0, T, device=device)
        h = self.token_emb(x) + self.pos_emb(pos)[None, :, :]
        h = self.drop(h)

        if attn_mask is None:
            attn_mask = self.causal_mask(T, device)

        for blk in self.blocks:
            h = blk(h, attn_mask)
        h = self.norm(h)
        return self.head(h)


# ============================================================
# Transformer block for the traditional model
# ============================================================
class TransformerBlockClassic(nn.Module):
    """
    Classic Transformer block with:
      - LayerNorm (pre-norm)
      - MultiHeadAttention
      - MLP (2-layer GELU)
    """

    def __init__(self, dim, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # Self-attention with residual
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)

        # Feed-forward with residual
        h = self.ln2(x)
        x = x + self.ff(h)
        return x
