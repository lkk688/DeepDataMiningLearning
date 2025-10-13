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

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Full Transformer (Encoder–Decoder architecture)
# ============================================================
class FullTransformer(nn.Module):
    """
    Full Transformer model (Encoder–Decoder) as in "Attention Is All You Need" (Vaswani et al., 2017).

    This model supports sequence-to-sequence tasks such as:
        - machine translation
        - summarization
        - question answering
        - text-to-text generation

    Key components:
      - Encoder: processes input sequence (source)
      - Decoder: generates output sequence (target)
      - Cross-attention: decoder attends to encoder outputs
      - Absolute positional embeddings
      - LayerNorm + GELU activation
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        share_embeddings: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads

        # ------------------------------
        # 1. Embeddings + Positional Encoding
        # ------------------------------
        self.src_emb = nn.Embedding(src_vocab_size, dim)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.dropout = nn.Dropout(dropout)

        # Optionally share source & target embeddings
        if share_embeddings and src_vocab_size == tgt_vocab_size:
            self.tgt_emb.weight = self.src_emb.weight

        # ------------------------------
        # 2. Encoder & Decoder Stacks
        # ------------------------------
        encoder_layer = TransformerEncoderLayerClassic(dim, n_heads, ff_dim, dropout)
        decoder_layer = TransformerDecoderLayerClassic(dim, n_heads, ff_dim, dropout)
        self.encoder = nn.ModuleList([encoder_layer for _ in range(n_layers)])
        self.decoder = nn.ModuleList([decoder_layer for _ in range(n_layers)])

        # ------------------------------
        # 3. Final LayerNorm and Output Projection
        # ------------------------------
        self.norm = nn.LayerNorm(dim)
        self.output_head = nn.Linear(dim, tgt_vocab_size, bias=False)

        self._reset_parameters()

    # ------------------------------------------------------------
    # Parameter initialization
    # ------------------------------------------------------------
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------
    # Helper: create masks
    # ------------------------------------------------------------
    @staticmethod
    def generate_square_subsequent_mask(sz: int, device):
        """Generate a causal mask for the decoder (prevent future attention)."""
        mask = torch.triu(torch.ones(sz, sz, device=device) * float("-inf"), diagonal=1)
        return mask

    # ------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------
    def forward(self, src, tgt):
        """
        Args:
            src: [B, S] source token indices
            tgt: [B, T] target token indices (teacher-forcing during training)
        Returns:
            logits: [B, T, V_tgt] unnormalized token scores
        """
        B, S = src.shape
        _, T = tgt.shape
        device = src.device

        # ---- Step 1: Embeddings + positional encoding ----
        pos_src = torch.arange(0, S, device=device)
        pos_tgt = torch.arange(0, T, device=device)
        src_emb = self.dropout(self.src_emb(src) + self.pos_emb(pos_src)[None, :, :])
        tgt_emb = self.dropout(self.tgt_emb(tgt) + self.pos_emb(pos_tgt)[None, :, :])

        # ---- Step 2: Encoder ----
        enc_out = src_emb
        for layer in self.encoder:
            enc_out = layer(enc_out)

        # ---- Step 3: Decoder ----
        tgt_mask = self.generate_square_subsequent_mask(T, device)
        dec_out = tgt_emb
        for layer in self.decoder:
            dec_out = layer(dec_out, enc_out, tgt_mask=tgt_mask)

        # ---- Step 4: Normalize & project to vocabulary ----
        dec_out = self.norm(dec_out)
        logits = self.output_head(dec_out)  # [B, T, V_tgt]

        return logits
    
# ============================================================
# Encoder Layer
# ============================================================
class TransformerEncoderLayerClassic(nn.Module):
    def __init__(self, dim, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # Self-attention
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feed-forward
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


# ============================================================
# Decoder Layer
# ============================================================
class TransformerDecoderLayerClassic(nn.Module):
    def __init__(self, dim, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None):
        """
        Args:
            tgt: [B, T, D] target embeddings
            memory: [B, S, D] encoder outputs
            tgt_mask: [T, T] causal mask
        """
        # Masked self-attention (causal)
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention: decoder attends to encoder outputs
        tgt2, _ = self.cross_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feed-forward
        tgt2 = self.linear2(self.dropout(F.gelu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


# ============================================================
# PyTorch Built-in Transformer (Encoder–Decoder)
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class PyTorchTransformer(nn.Module):
    """
    Wrapper around torch.nn.Transformer for seq2seq learning.

    Uses standard components:
      - token embeddings + positional encodings
      - nn.Transformer (Encoder + Decoder)
      - projection to vocab logits

    Suitable for translation and general seq2seq tasks.
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        dim=512,
        n_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        ff_dim=2048,
        dropout=0.1,
        max_seq_len=512,
        share_embeddings=False,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Embeddings
        self.src_emb = nn.Embedding(src_vocab_size, dim)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.dropout = nn.Dropout(dropout)

        # Core Transformer
        self.transformer = nn.Transformer(
            d_model=dim,
            nhead=n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.fc_out = nn.Linear(dim, tgt_vocab_size)
        if share_embeddings and src_vocab_size == tgt_vocab_size:
            self.tgt_emb.weight = self.src_emb.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt_in):
        """
        Args:
            src: [B, S] source tokens
            tgt_in: [B, T] target input tokens (shifted)
        Returns:
            logits: [B, T, V_tgt]
        """
        device = src.device
        B, S = src.shape
        _, T = tgt_in.shape

        # Add embeddings + positional encodings
        pos_src = torch.arange(0, S, device=device)
        pos_tgt = torch.arange(0, T, device=device)
        src_emb = self.dropout(self.src_emb(src) + self.pos_emb(pos_src)[None, :, :])
        tgt_emb = self.dropout(self.tgt_emb(tgt_in) + self.pos_emb(pos_tgt)[None, :, :])

        # Causal mask for decoder (no future tokens)
        tgt_mask = self.transformer.generate_square_subsequent_mask(T).to(device)

        # Forward through transformer
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)

        # Output projection
        logits = self.fc_out(out)
        return logits
    
# ============================================================
# Recurrent Models: Traditional RNN and LSTM Baselines
# ============================================================
# ============================================================
# Recurrent Language Model Base Class (RNN / LSTM / GRU)
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentLanguageModelBase(nn.Module):
    """
    Generic template for character/word-level language models
    using RNN / LSTM / GRU backends.

    Shared features:
      ✅ Embedding + recurrent backbone (defined by subclass)
      ✅ LayerNorm for stable training
      ✅ Residual connections
      ✅ Dropout + weight tying
      ✅ Compatible with Trainer/Evaluator
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.2,
        rnn_type: str = "lstm",  # "rnn", "gru", or "lstm"
        tie_weights: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_type = rnn_type.lower()

        # --- Embedding ---
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.drop = nn.Dropout(dropout)

        # --- Core RNN family selection ---
        if self.rnn_type == "rnn":
            self.recurrent = nn.RNN(
                embed_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                nonlinearity="tanh",
                batch_first=True,
            )
        elif self.rnn_type == "gru":
            self.recurrent = nn.GRU(
                embed_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
        elif self.rnn_type == "lstm":
            self.recurrent = nn.LSTM(
                embed_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unknown rnn_type: {self.rnn_type}")

        # --- Normalization and output ---
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        # Optional weight tying
        if tie_weights:
            if embed_dim != hidden_dim:
                raise ValueError("For weight tying, embed_dim must equal hidden_dim.")
            self.fc_out.weight = self.embed.weight

        self.init_weights()

    # ------------------------------------------------------------
    def init_weights(self):
        """Xavier initialization for weights, zeros for bias."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    # ------------------------------------------------------------
    def forward(self, x, hidden=None):
        """
        Forward pass through the recurrent LM.

        Args:
            x: [B, T] token indices
            hidden: optional hidden state (h or (h, c))
        Returns:
            logits: [B, T, V]
            new_hidden: recurrent hidden state
        """
        emb = self.drop(self.embed(x))         # [B, T, E]
        out, new_hidden = self.recurrent(emb, hidden)
        out = self.norm(out)
        out = self.drop(out)

        # Residual connection (if dims match)
        if emb.shape[-1] == out.shape[-1]:
            out = out + emb

        logits = self.fc_out(out)
        return logits, new_hidden

    # ------------------------------------------------------------
    def detach_hidden(self, hidden):
        """Detaches hidden state from previous computation graph."""
        if hidden is None:
            return None
        if isinstance(hidden, tuple):  # LSTM
            return tuple(h.detach() for h in hidden)
        else:  # RNN/GRU
            return hidden.detach()

# ============================================================
# Simple subclasses for clarity and configuration
# ============================================================
class RNNLanguageModel(RecurrentLanguageModelBase):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.2, tie_weights=True):
        super().__init__(vocab_size, embed_dim, hidden_dim, num_layers, dropout, rnn_type="rnn", tie_weights=tie_weights)


class LSTMLanguageModel(RecurrentLanguageModelBase):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.3, tie_weights=True):
        super().__init__(vocab_size, embed_dim, hidden_dim, num_layers, dropout, rnn_type="lstm", tie_weights=tie_weights)


class GRULanguageModel(RecurrentLanguageModelBase):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.2, tie_weights=True):
        super().__init__(vocab_size, embed_dim, hidden_dim, num_layers, dropout, rnn_type="gru", tie_weights=tie_weights)
        
# ============================================================
# MODEL INITIALIZATION FUNCTION (clean version)
# ============================================================
def build_model(model_type, data, args):
    """
    Dynamically build the model according to the selected architecture.

    Args:
        model_type (str): Model architecture type.
        data: DataModule or similar providing vocab sizes.
        args: CLI arguments with model hyperparameters.

    Returns:
        model (nn.Module): The initialized model.
        hf_mode (bool): Whether this model uses Hugging Face's forward signature.
    """
    hf_mode = False

    # --- Modern Transformer (decoder-only) ---
    if model_type == "TransformerLM":
        print("🚀 Initializing modern TransformerLM (RMSNorm + RoPE + SwiGLU)")
        cfg = ModelConfig(
            vocab_size=data.vocab_size,
            dim=args.dim,
            n_layers=args.layers,
            n_heads=args.heads,
            max_seq_len=args.seq_len,
            rope=args.rope,
            dropout=0.1,
        )
        model = TransformerLM(cfg)

    # --- Traditional Transformer (GPT-2 style) ---
    elif model_type == "TraditionalTransformerLM":
        print("🧩 Initializing Traditional Transformer (LayerNorm + GELU + AbsPos)")
        model = TraditionalTransformerLM(
            vocab_size=data.vocab_size,
            dim=args.dim,
            n_layers=args.layers,
            n_heads=args.heads,
            ff_dim=args.dim * 4,
            dropout=0.1,
            max_seq_len=args.seq_len,
        )

    # --- Full Transformer (Encoder–Decoder) ---
    elif model_type == "FullTransformer":
        print("🧠 Initializing Full Transformer (Encoder–Decoder)")
        model = FullTransformer(
            src_vocab_size=data.tokenizer_src.vocab_size,
            tgt_vocab_size=data.tokenizer_tgt.vocab_size,
            dim=args.dim,
            n_layers=args.layers,
            n_heads=args.heads,
            ff_dim=args.dim * 4,
            dropout=0.1,
            max_seq_len=args.seq_len,
        )

    # --- RNN-based language model ---
    elif model_type == "RNN":
        print("🧠 Initializing traditional RNN Language Model...")
        model = RNNLanguageModel(
            vocab_size=data.vocab_size,
            embed_dim=args.dim,
            hidden_dim=args.dim,
            num_layers=args.layers,
            dropout=0.2,
        )

    # --- LSTM-based language model ---
    elif model_type == "LSTM":
        print("🧠 Initializing LSTM Language Model...")
        model = LSTMLanguageModel(
            vocab_size=data.vocab_size,
            embed_dim=args.dim,
            hidden_dim=args.dim,
            num_layers=args.layers,
            dropout=0.2,
        )

    # --- Hugging Face pretrained model ---
    elif model_type == "hf":
        print(f"🤗 Loading Hugging Face model: {args.hf_model_name}")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.hf_model_name)
        hf_mode = True

    else:
        raise ValueError(f"❌ Unknown model_type: {model_type}")

    return model, hf_mode