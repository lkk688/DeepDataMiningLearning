# Mixed (B14) with a different random seed for C2 multi-seed analysis.
_base_ = ['./B14_waymo_nuscenes_mixed.py']
randomness = dict(seed=20260524, deterministic=False)
