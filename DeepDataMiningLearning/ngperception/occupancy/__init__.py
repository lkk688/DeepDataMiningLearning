"""
ngperception.occupancy — 3D semantic occupancy prediction (Occ3D-nuScenes).

Predict a dense voxel grid of semantics around the ego vehicle. This is the downstream
step *above* depth: depth gives per-pixel range, occupancy gives a full 3D scene. We
start with a **depth->occupancy geometric baseline** (lift monocular metric depth from the
6 surround cameras into the voxel grid) and score it against the Occ3D GT — a measured
answer to "how far does a depth foundation model alone get you toward learned occupancy?".
"""
