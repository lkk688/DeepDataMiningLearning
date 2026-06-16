# Mixed+PL (B18) with a third random seed.
_base_ = ['./B18_pseudo_label_v14.py']
randomness = dict(seed=4242, deterministic=False)
