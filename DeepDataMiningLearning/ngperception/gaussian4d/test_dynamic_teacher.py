"""Quick check: does dynamic/static separation improve the label-free pseudo-label's agreement with
Occ3D-GT (esp. foreground) vs the no-dynamic voxel teacher? Teacher-vs-GT mIoU/geo/tail/FG."""
import numpy as np, os
from nuscenes import NuScenes
from .teachers.dynamic_teacher import DynamicOccTeacher
from .teachers.voxel_teacher import VoxelTeacher
from ..occupancy.datasets import Occ3DNuScenesDataset
from ..occupancy.evaluator import OccupancyEvaluator, OCC3D_CLASSES
from .teachers.base import TAIL_CLASSES

ROOT = "/data/rnd-liu/Datasets/nuScenes"
nusc = NuScenes(version="v1.0-trainval", dataroot=ROOT + "/v1.0-trainval", verbose=False)
cache = ROOT + "/labelgen_cache"
occ = Occ3DNuScenesDataset(ROOT + "/v1.0-trainval/gts", scenes=None)
items = [(sc, tok, lp) for sc, tok, lp in occ.items if os.path.isfile(os.path.join(cache, tok + ".npz"))][:12]
FG = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for name, T in [("voxel10(no-dyn)", VoxelTeacher(nusc, cache, sweeps=10)),
                ("DynamicOcc(dyn+gtbox)", DynamicOccTeacher(nusc, cache, sweeps=10, boxes="gt"))]:
    ev = OccupancyEvaluator()
    for sc, tok, lp in items:
        tgt = T(tok); g = np.load(lp)
        ev.add(tgt.semantics, g["semantics"].astype(np.uint8), g["mask_camera"].astype(bool))
    s = ev.summarize(verbose=False)
    tail = np.mean([s["per_class"][OCC3D_CLASSES[c]] for c in TAIL_CLASSES])
    fg = np.mean([s["per_class"][OCC3D_CLASSES[c]] for c in FG])
    print("%-24s mIoU=%.3f geo=%.3f tail=%.3f FG=%.3f" % (name, s["mIoU"], s["geo_IoU"], tail, fg))
