"""One decoder-only Transformer whose every "modern trick" is a config switch.

With all golf knobs off (the ModelConfig defaults) this is a plain pre-norm GPT:
    x = tok_emb(idx) (+ pos_emb)
    for each block:  x = x + Attn(Norm(x));  x = x + MLP(Norm(x))
    logits = Norm(x) @ W_out

Each parameter-golf technique is implemented in a few lines next to a comment that says what it
does and where it came from, so students can switch it on in a YAML file and measure it:

  qk_norm / qk_gain_init  RMS-normalise q,k (+ learnable per-head temperature)      golf baseline
  logit_softcap           cap * tanh(logits / cap)                                  golf baseline
  layer_scales, resid_mix per-channel branch scales; mix stream with x0           golf baseline
  unet_skips              encoder-half outputs feed mirrored decoder layers         golf baseline
  mlp=relu2/leaky_relu2   squared (leaky) ReLU MLP                                  golf baseline / record 03-23
  n_kv_heads < n_heads    grouped-query attention                                   golf baseline
  rope_dims               partial RoPE (rotate only the first k dims of each head)  record 03-21
  ln_scale                normed input of layer i scaled by 1/sqrt(i+1)             record 03-21
  xsa_last_n              exclusive self-attention: remove own-value component      record 03-20
  smear_gate              gated blend with the previous token's embedding           record 03-19
  bigram_buckets          hashed (prev, cur) bigram embedding                       record 03-19
  value_residual          mix layer-0 values into later layers                      ResFormer
  recur_layers            depth recurrence: re-run a band of layers (weight reuse)  records 03-31+
  parallel_from           attention and MLP read the same input (GPT-J/PaLM style)  records 03-31+
  attn_gate               per-head sigmoid gate on attention output                 records 04-19+
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================ building blocks
def fake_quant_ste(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-row symmetric fake quantization with a straight-through estimator (QAT)."""
    qmax = 2 ** (bits - 1) - 1
    scale = (w.detach().abs().amax(dim=1, keepdim=True) / qmax).clamp_min(1e-8)
    wq = torch.clamp(torch.round(w / scale), -qmax, qmax) * scale
    return w + (wq - w).detach()          # forward: quantized, backward: identity


class Linear(nn.Linear):
    """nn.Linear that keeps fp32 master weights and can fake-quantize them (QAT)."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.qat_bits = 0

    def forward(self, x):
        w = fake_quant_ste(self.weight, self.qat_bits) if self.qat_bits else self.weight
        return F.linear(x, w.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x):
        x = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)
        return x * self.weight.to(x.dtype) if self.weight is not None else x


def make_norm(kind: str, dim: int, bias: bool):
    if kind == "layernorm":
        return nn.LayerNorm(dim, bias=bias)
    if kind == "rmsnorm":
        return RMSNorm(dim)
    raise ValueError(f"unknown norm {kind!r}")


def rms(x):  # parameter-free RMS normalisation
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype)


class Rotary(nn.Module):
    """RoPE, half-split convention, cos/sin cached per (length, device)."""

    def __init__(self, rot_dim: int, base: float):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, rot_dim, 2, dtype=torch.float32) / rot_dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self.rot_dim = rot_dim
        self._cache = (0, None, None, None)

    def forward(self, T: int, device):
        n, dev, cos, sin = self._cache
        if n != T or dev != device:
            t = torch.arange(T, device=device, dtype=torch.float32)
            f = torch.outer(t, self.inv_freq.to(device))
            cos, sin = f.cos()[None, None], f.sin()[None, None]
            self._cache = (T, device, cos, sin)
        return cos, sin


def apply_rope(x, cos, sin, rot_dim: int):
    """Rotate the first `rot_dim` dims of every head; pass the rest through (partial RoPE)."""
    xr, xp = x[..., :rot_dim], x[..., rot_dim:]
    h = rot_dim // 2
    x1, x2 = xr[..., :h], xr[..., h:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    out = torch.cat((x1 * cos + x2 * sin, x2 * cos - x1 * sin), dim=-1)
    return torch.cat((out, xp), dim=-1) if xp.numel() else out


# ================================================================================= attention
class Attention(nn.Module):
    def __init__(self, cfg, layer: int):
        super().__init__()
        D, H = cfg.dim, cfg.n_heads
        KV = cfg.n_kv_heads or H
        assert D % H == 0 and H % KV == 0, "dim % n_heads and n_heads % n_kv_heads must be 0"
        self.H, self.KV, self.hd = H, KV, D // H
        self.c_q = Linear(D, D, bias=cfg.bias)
        self.c_k = Linear(D, KV * self.hd, bias=cfg.bias)
        self.c_v = Linear(D, KV * self.hd, bias=cfg.bias)
        self.proj = Linear(D, D, bias=cfg.bias)
        self.proj._is_out_proj = True
        self.dropout = cfg.dropout
        self.qk_norm = cfg.qk_norm
        self.q_gain = nn.Parameter(torch.full((H,), cfg.qk_gain_init)) if cfg.qk_gain_init > 0 else None
        self.rope = None
        if cfg.pos == "rope":
            rd = cfg.rope_dims if 0 < cfg.rope_dims < self.hd else self.hd
            self.rope = Rotary(rd, cfg.rope_base)
        # XSA (exclusive self-attention): the output of a head partly just copies the token's own
        # value vector; XSA projects that component out so attention must bring *new* information.
        self.xsa = cfg.xsa_last_n > 0 and layer >= cfg.n_layers - cfg.xsa_last_n
        # Value residual (ResFormer): v_l <- a*v_0 + b*v_l, lets late layers reuse early token info.
        self.v_mix = nn.Parameter(torch.tensor([0.5, 0.5])) if cfg.value_residual and layer > 0 else None
        # Gated attention: out_h <- 2*sigmoid(x W_g)_h * out_h  (starts as identity because W_g = 0)
        self.gate = None
        if cfg.attn_gate:
            self.gate = Linear(D, H, bias=False)
            nn.init.zeros_(self.gate.weight)
            self.gate._no_init = True

    def forward(self, x, v0=None):
        B, T, D = x.shape
        q = self.c_q(x).view(B, T, self.H, self.hd).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.KV, self.hd).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.KV, self.hd).transpose(1, 2)
        v_raw = v
        if self.v_mix is not None and v0 is not None:
            v = self.v_mix[0].to(v.dtype) * v0 + self.v_mix[1].to(v.dtype) * v
        if self.qk_norm:
            q, k = rms(q), rms(k)
        if self.rope is not None:
            cos, sin = self.rope(T, x.device)
            q = apply_rope(q, cos, sin, self.rope.rot_dim)
            k = apply_rope(k, cos, sin, self.rope.rot_dim)
        if self.q_gain is not None:
            q = q * self.q_gain.to(q.dtype)[None, :, None, None]
        if self.KV != self.H:                         # GQA: share each k/v head across a group
            rep = self.H // self.KV
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        if self.xsa:
            vn = v / (v.norm(dim=-1, keepdim=True) + 1e-6)
            y = y - (y * vn).sum(-1, keepdim=True) * vn
        if self.gate is not None:
            y = y * (2 * torch.sigmoid(self.gate(x))).transpose(1, 2)[..., None]
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(y), v_raw


# ======================================================================================= MLP
class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D = cfg.dim
        self.kind = cfg.mlp
        if self.kind == "swiglu":           # 2/3 factor keeps params equal to a 2-matrix MLP
            hidden = int(cfg.mlp_mult * D * 2 / 3)
            self.fc = Linear(D, 2 * hidden, bias=cfg.bias)
        else:
            hidden = int(cfg.mlp_mult * D)
            self.fc = Linear(D, hidden, bias=cfg.bias)
        self.proj = Linear(hidden, D, bias=cfg.bias)
        self.proj._is_out_proj = True

    def forward(self, x):
        h = self.fc(x)
        if self.kind == "gelu":
            h = F.gelu(h)
        elif self.kind == "relu2":            # modded-nanogpt / golf baseline
            h = F.relu(h).square()
        elif self.kind == "leaky_relu2":      # golf record 2026-03-23: keeps a gradient for x<0
            h = F.leaky_relu(h, 0.5).square()
        elif self.kind == "swiglu":
            a, b = h.chunk(2, dim=-1)
            h = F.silu(a) * b
        else:
            raise ValueError(f"unknown mlp {self.kind!r}")
        return self.proj(h)


# ===================================================================================== block
class Block(nn.Module):
    def __init__(self, cfg, layer: int):
        super().__init__()
        D = cfg.dim
        self.attn_norm = make_norm(cfg.norm, D, cfg.bias)
        self.mlp_norm = make_norm(cfg.norm, D, cfg.bias)
        self.attn = Attention(cfg, layer)
        self.mlp = MLP(cfg)
        self.drop = nn.Dropout(cfg.dropout)
        self.attn_scale = nn.Parameter(torch.ones(D)) if cfg.layer_scales else None
        self.mlp_scale = nn.Parameter(torch.ones(D)) if cfg.layer_scales else None
        self.resid_mix = nn.Parameter(torch.stack([torch.ones(D), torch.zeros(D)])) if cfg.resid_mix else None
        self.in_scale = 1.0 / math.sqrt(layer + 1) if cfg.ln_scale else 1.0
        self.parallel = 0 <= cfg.parallel_from <= layer

    def _scale(self, s, h):
        return h if s is None else s.to(h.dtype) * h

    def forward(self, x, x0, v0=None):
        if self.resid_mix is not None:
            m = self.resid_mix.to(x.dtype)
            x = m[0] * x + m[1] * x0
        a, v = self.attn(self.attn_norm(x) * self.in_scale, v0)
        if self.parallel:                     # both branches read the same x
            m = self.mlp(self.mlp_norm(x) * self.in_scale)
            return x + self.drop(self._scale(self.attn_scale, a)) + self.drop(self._scale(self.mlp_scale, m)), v
        x = x + self.drop(self._scale(self.attn_scale, a))
        x = x + self.drop(self._scale(self.mlp_scale, self.mlp(self.mlp_norm(x) * self.in_scale)))
        return x, v


# ========================================================================== input add-ons
class SmearGate(nn.Module):
    """x_t <- (1-g) x_t + g x_{t-1}, g = sigmoid(per-channel param): a free 'bigram' look-back."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        g = torch.sigmoid(self.gate.to(x.dtype))
        prev = F.pad(x[:, :-1], (0, 0, 1, 0))
        return (1 - g) * x + g * prev


class BigramHash(nn.Module):
    """Embedding of hash(prev_token, token) into `buckets` rows, projected to model dim.
    Gives the model a cheap lookup for frequent token pairs (zero-init projection)."""

    def __init__(self, buckets: int, bdim: int, dim: int):
        super().__init__()
        self.buckets = buckets
        self.emb = nn.Embedding(buckets, bdim)
        nn.init.normal_(self.emb.weight, std=0.01)
        self.emb._no_init = True
        self.proj = Linear(bdim, dim, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.proj._no_init = True
        self.scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, idx):
        prev = F.pad(idx[:, :-1], (1, 0), value=0)
        h = torch.bitwise_xor(idx * 36313, prev * 27191) % (self.buckets - 1)
        h[:, 0] = self.buckets - 1                       # position 0 has no predecessor
        return self.proj(self.emb(h)) * self.scale.to(self.proj.weight.dtype)


# ======================================================================================= GPT
class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        V, D, L = cfg.vocab_size, cfg.dim, cfg.n_layers
        self.tok_emb = nn.Embedding(V, D)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, D) if cfg.pos == "learned" else None
        self.bigram = BigramHash(cfg.bigram_buckets, cfg.bigram_dim, D) if cfg.bigram_buckets else None
        self.smear = SmearGate(D) if cfg.smear_gate else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(L)])
        self.final_norm = make_norm(cfg.norm, D, cfg.bias)
        self.lm_head = None if cfg.tie_embeddings else Linear(D, V, bias=False)

        # U-Net: layer i < L//2 is an "encoder" layer; decoder layer L//2 + j receives the output
        # of encoder layer L//2 - 1 - j through a learnable per-channel weight.
        self.skip_src: dict[int, int] = {}
        if cfg.unet_skips:
            n_enc = L // 2
            for j in range(min(n_enc, L - n_enc)):
                self.skip_src[n_enc + j] = n_enc - 1 - j
            self.skip_weights = nn.Parameter(torch.ones(len(self.skip_src), D))
        # Depth recurrence: the executed layer order, e.g. 0..10 with [3,4,5] looped once more
        # -> 0 1 2 3 4 5 3 4 5 6 ... 10. Switched on by the trainer at recur_start_frac.
        self.recur_active = bool(cfg.recur_layers) and cfg.recur_start_frac <= 0
        self._init_weights()

    # ------------------------------------------------------------------------------- init
    def _init_weights(self):
        cfg = self.cfg
        std = cfg.init_std
        out_std = std / math.sqrt(2 * cfg.n_layers)        # GPT-2: scale residual-branch outputs
        for m in self.modules():
            if getattr(m, "_no_init", False):
                continue
            if isinstance(m, nn.Linear):
                if getattr(m, "_is_out_proj", False):
                    if cfg.zero_init_proj:
                        nn.init.zeros_(m.weight)     # every block starts as the identity map
                    else:
                        nn.init.normal_(m.weight, std=out_std)
                else:
                    nn.init.normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.tok_emb.weight, std=cfg.embed_init_std or std)
        if self.pos_emb is not None:
            nn.init.normal_(self.pos_emb.weight, std=std)

    def layer_schedule(self) -> list[int]:
        L = list(range(self.cfg.n_layers))
        if not (self.recur_active and self.cfg.recur_layers):
            return L
        band = list(self.cfg.recur_layers)
        last = band[-1]
        return L[: last + 1] + band * self.cfg.recur_times + L[last + 1:]

    # ---------------------------------------------------------------------------- forward
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        if self.bigram is not None:
            x = x + self.bigram(idx).to(x.dtype)
        if self.cfg.embed_norm:
            x = rms(x)
        if self.smear is not None:
            x = self.smear(x)
        x = self.drop(x)
        x0, v0 = x, None
        enc_out: dict[int, torch.Tensor] = {}
        visited: set[int] = set()
        skip_idx = {dst: k for k, dst in enumerate(self.skip_src)}
        for i in self.layer_schedule():
            if i in self.skip_src and i not in visited:
                x = x + self.skip_weights[skip_idx[i]].to(x.dtype) * enc_out[self.skip_src[i]]
            x, v = self.blocks[i](x, x0, v0)
            if v0 is None:
                v0 = v
            if self.skip_src:
                enc_out[i] = x
            visited.add(i)
        x = self.final_norm(x)
        w = self.tok_emb.weight if self.lm_head is None else self.lm_head.weight
        logits = F.linear(x, w.to(x.dtype)).float()
        if self.cfg.logit_softcap > 0:
            c = self.cfg.logit_softcap
            logits = c * torch.tanh(logits / c)
        return logits

    # -------------------------------------------------------------------------- utilities
    def matrix_params(self):
        """2-D hidden weights (Muon-eligible): everything 2-D inside the blocks."""
        return [p for n, p in self.blocks.named_parameters()
                if p.ndim == 2 and min(p.shape) > 2 and "resid_mix" not in n]

    def set_qat(self, bits: int):
        for m in self.blocks.modules():
            if isinstance(m, Linear):
                m.qat_bits = bits
