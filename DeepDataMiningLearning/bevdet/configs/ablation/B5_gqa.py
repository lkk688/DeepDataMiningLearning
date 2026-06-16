# =====================================================================
# B5 = B4 + Multi-Head Grouped-Query Cross-Attention (GQA-CA).  ⭐
#
# THE PAPER'S HEADLINE CLAIM.
#
# Adds multi-head structure with GQA grouping to the BEV-query
# cross-attention. With num_heads=8, num_kv_groups=4 we get 4× K/V
# memory reduction vs full MHA at the cost of +0.04 M parameters,
# while gaining multi-head expressivity for per-head specialization.
#
# This is the only row that introduces our novel contribution:
# GQA applied to BEV view transform for the first time.
# Prior cross-attention BEV view transforms (BEVFormer, PETR, DETR3D,
# Tesla AI Day 2021) all use single-head or pure MHA — none use GQA.
# =====================================================================

_base_ = ['./B4_auxbev.py']

model = dict(
    view_transform=dict(
        # Multi-Head + GQA configuration. num_heads must divide in_channels (128 here).
        num_heads=8,            # H_q query heads
        num_kv_groups=4,        # G groups → H_kv = 8/4 = 2 K/V heads
        # Note: keeping attn_chunk=8192 from B1; can sweep separately.
    ),
)
