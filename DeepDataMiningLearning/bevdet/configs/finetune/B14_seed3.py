# Mixed (B14) with a third random seed.
_base_ = ['./B14_waymo_nuscenes_mixed.py']
randomness = dict(seed=4242, deterministic=False)
