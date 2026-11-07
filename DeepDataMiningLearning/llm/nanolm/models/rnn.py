"""Recurrent language models: vanilla RNN, GRU, LSTM -- plus an LSTM and a GRU written from scratch.

All models share the nanolm interface:  logits = model(idx)   with idx [B, T] -> logits [B, T, V].
Each training window starts from a zero hidden state (truncated BPTT of length seq_len), which
keeps RNNs and Transformers on exactly the same data pipeline and evaluation.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMScratch(nn.Module):
    """A single-layer LSTM written out gate by gate (for reading, not for speed).

        i = sigma(W_i [x, h])    input gate      -- how much new information to write
        f = sigma(W_f [x, h])    forget gate     -- how much of the old cell to keep
        g = tanh (W_g [x, h])    candidate cell
        o = sigma(W_o [x, h])    output gate
        c' = f * c + i * g       <- the additive cell path is why gradients survive long spans
        h' = o * tanh(c')
    """

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.W = nn.Linear(in_dim + hidden, 4 * hidden)
        with torch.no_grad():                      # forget-gate bias = 1: remember by default
            self.W.bias.zero_()
            self.W.bias[hidden: 2 * hidden] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x [B, T, D]
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hidden)
        c = x.new_zeros(B, self.hidden)
        outs = []
        for t in range(T):
            i, f, g, o = self.W(torch.cat([x[:, t], h], dim=-1)).chunk(4, dim=-1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            outs.append(h)
        return torch.stack(outs, dim=1)


class GRUScratch(nn.Module):
    """A single-layer GRU written out gate by gate (for reading, not for speed). Matches nn.GRU exactly.

        r  = sigma(W_ir x + b_ir + W_hr h + b_hr)          reset gate   -- how much of the past to use
        z  = sigma(W_iz x + b_iz + W_hz h + b_hz)          update gate  -- how much of the old state to keep
        n  = tanh (W_in x + b_in + r * (W_hn h + b_hn))    candidate state
        h' = (1 - z) * n + z * h                           <- interpolation, the GRU's additive path

    Unlike LSTMScratch, input and hidden projections stay separate: r multiplies the hidden projection
    *including its bias* b_hn ("reset after", as in nn.GRU / cuDNN). reset_after=False gives the
    original Cho et al. (2014) cell, n = tanh(W_in x + b_in + W_hn (r * h) + b_hn), which does not.
    Weight rows are stacked [r | z | n], the layout of nn.GRU's weight_ih_l{k} / weight_hh_l{k}.
    """

    def __init__(self, in_dim: int, hidden: int, reset_after: bool = True):
        super().__init__()
        self.hidden = hidden
        self.reset_after = reset_after
        self.W_x = nn.Linear(in_dim, 3 * hidden)
        self.W_h = nn.Linear(hidden, 3 * hidden)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:   # x [B, T, D]
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hidden) if h is None else h
        xr, xz, xn = self.W_x(x).chunk(3, dim=-1)          # input projections of all steps at once
        Wr, Wz, Wn = self.W_h.weight.chunk(3, dim=0)
        br, bz, bn = self.W_h.bias.chunk(3, dim=0)
        outs = []
        for t in range(T):
            r = torch.sigmoid(xr[:, t] + h @ Wr.T + br)
            z = torch.sigmoid(xz[:, t] + h @ Wz.T + bz)
            if self.reset_after:
                n = torch.tanh(xn[:, t] + r * (h @ Wn.T + bn))
            else:
                n = torch.tanh(xn[:, t] + (r * h) @ Wn.T + bn)
            h = (1 - z) * n + z * h
            outs.append(h)
        return torch.stack(outs, dim=1)

    @torch.no_grad()
    def load_from(self, gru: nn.GRU, layer: int = 0) -> "GRUScratch":
        """Copy layer `layer` of an nn.GRU into this cell."""
        self.W_x.weight.copy_(getattr(gru, f"weight_ih_l{layer}"))
        self.W_x.bias.copy_(getattr(gru, f"bias_ih_l{layer}"))
        self.W_h.weight.copy_(getattr(gru, f"weight_hh_l{layer}"))
        self.W_h.bias.copy_(getattr(gru, f"bias_hh_l{layer}"))
        return self


class RecurrentLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        V, D = cfg.vocab_size, cfg.dim
        H = cfg.rnn_hidden or D
        self.cfg = cfg
        self.embed = nn.Embedding(V, D)
        self.drop = nn.Dropout(cfg.dropout)
        kind = cfg.arch
        if kind in ("lstm_scratch", "gru_scratch"):
            cell = LSTMScratch if kind == "lstm_scratch" else GRUScratch
            layers = [cell(D if i == 0 else H, H) for i in range(cfg.n_layers)]
            self.rnn = nn.Sequential(*layers)
        else:
            cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[kind]
            self.rnn = cls(D, H, num_layers=cfg.n_layers, batch_first=True,
                           dropout=cfg.dropout if cfg.n_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(H)
        self.head = nn.Linear(H, V, bias=False)
        if cfg.tie_embeddings and H == D:
            self.head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=cfg.init_std)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.embed(idx))
        out = self.rnn(x)
        if isinstance(out, tuple):        # nn.RNN/GRU/LSTM return (output, hidden)
            out = out[0]
        return self.head(self.drop(self.norm(out))).float()
