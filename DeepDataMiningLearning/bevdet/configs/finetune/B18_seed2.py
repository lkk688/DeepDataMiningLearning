# Mixed+PL (B18) with a different random seed for C2.
_base_ = ['./B18_pseudo_label_v14.py']
randomness = dict(seed=20260524, deterministic=False)
