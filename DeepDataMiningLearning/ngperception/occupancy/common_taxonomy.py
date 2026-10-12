"""Per-sample occupancy counts in Qwen-Drive's 10-class taxonomy (the common label space for comparing
Qwen-Drive's occ head against Occ3D-trained models). Classes scored: the 8 reachable non-empty classes;
`czone_sign` (unreachable from nuScenes GT) and `empty` (free) are excluded from the class mean, and
geometric IoU is occupied (!= empty) vs free. Same .npz layout as OccupancyEvaluator.dump()."""
from __future__ import annotations
import json
import numpy as np

QD_CLASSES = ('vehicle', 'czone_sign', 'bicycle', 'generic_object', 'pedestrian',
              'traffic_cone', 'barrier', 'driveable', 'background', 'empty')
QD_EMPTY = 9
SCORED = [0, 2, 3, 4, 5, 6, 7, 8]   # all but czone_sign(1) and empty(9)


class TaxonomyCounter:
    def __init__(self):
        self.rows = []

    def add(self, pred, gt, mask):
        sel = mask.astype(bool); p, g = pred[sel].astype(np.int64), gt[sel].astype(np.int64)
        row = np.zeros((len(SCORED) + 1, 3), np.int64)
        for k, c in enumerate(SCORED):
            pc, gc = p == c, g == c
            row[k] = (np.sum(pc & gc), np.sum(pc & ~gc), np.sum(~pc & gc))
        po, go = p != QD_EMPTY, g != QD_EMPTY
        row[-1] = (np.sum(po & go), np.sum(po & ~go), np.sum(~po & go))
        self.rows.append(row)

    def dump(self, path, tokens, meta=None):
        assert len(tokens) == len(self.rows)
        np.savez_compressed(path, tokens=np.asarray(tokens), counts=np.stack(self.rows),
                            classes=np.asarray([QD_CLASSES[c] for c in SCORED]),
                            meta=json.dumps({**(meta or {}), 'taxonomy': 'qwen_drive_10'}))
