# =====================================================================
# B5M = B4 + Full Multi-Head Cross-Attention (MHA, no GQA grouping).
#
# Identical to B5 (GQA) except num_kv_groups=1 → H_kv = H_q = 8.
# Used to disentangle TWO effects:
#   (B4 → B5M)  Single-head → MHA: does multi-head help accuracy at all?
#   (B5M → B5)  MHA → GQA: does GQA preserve MHA's accuracy with lower
#                          K/V memory?
#
# This row is critical for the paper's GQA claim. Without B5M, we cannot
# distinguish "GQA reduces memory" from "single-head was sufficient".
# =====================================================================

_base_ = ['./B4_auxbev.py']

model = dict(
    view_transform=dict(
        # Full MHA: 8 query heads AND 8 K/V heads (one K/V head per Q head).
        # Maximally expressive multi-head attention; highest K/V memory cost.
        num_heads=8,
        num_kv_groups=1,
    ),
)
