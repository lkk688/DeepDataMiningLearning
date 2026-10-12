"""Occ3D-nuScenes (18 classes) <-> Qwen-Drive (10 classes): the common ruler.

WHY THIS FILE IS THE PREREQUISITE FOR THE HEADROOM MEASUREMENT
--------------------------------------------------------------
`QWEN_DRIVE_LIDAR_TOKENS.md` §7 task (b) asks how much metric 3D a language-level
representation actually retains, against a real LiDAR-camera detector. Neither LVLDrive nor
VLGA answers it, and the reason it is unanswered is that **there is no common ruler**:
Qwen-Drive predicts a 10-class occupancy grid, every other experiment in this project uses
Occ3D-nuScenes' 18, and LVLDrive reports BEV IoU on a grounding task where the 2D box is
handed to the model. Building the ruler is the contribution; the (entirely expected)
conclusion that a VLM is worse than BEVFusion is not.

THE ONE PIECE OF LUCK: THE GRIDS ARE ALREADY IDENTICAL
------------------------------------------------------
Occ3D-nuScenes: 200x200x16 over [-40,-40,-1, 40,40,5.4], 0.4 m, **ego** frame, (X,Y,Z).
Qwen-Drive nuScenes occ (`configuration_perception.py` + `_normalized_occ_grid`):
`pc_range` [-40,-40,-1, 40,40,5.4]; target_w = 80/0.4 = 200, target_h = 200,
target_d = `occ_pillar_h` = 16, so dz = 6.4/16 = 0.4 m; output permuted to (X,Y,Z).
**Voxel-identical.** No resampling, no interpolation, no alignment error to argue about --
only the labels differ. `assert_grid_compatible()` states this as a checkable claim.

DIRECTION OF THE MAPPING, AND WHY IT MATTERS MORE THAN THE MAPPING
------------------------------------------------------------------
The 18 -> 10 map is **lossy, and lossy in a direction that flatters Qwen-Drive**:

* `car`, `truck`, `bus`, `trailer`, `construction_vehicle` all collapse to `vehicle`. A model
  that confuses a truck for a car is no longer charged for it.
* `other_flat`, `sidewalk`, `terrain`, `manmade`, `vegetation` all collapse to `background`.
* `motorcycle` has no honest target and is mapped to `bicycle` -- a real semantic error, kept
  only because leaving it unmapped would be worse.
* `czone_sign` has **no source class in nuScenes at all**; it can never appear in the GT, so
  any prediction of it is unconditionally wrong. Reported separately, never averaged in.

So mapping *our* GT down to their taxonomy and scoring only them would be an **upper bound on
their apparent accuracy**, not a comparison. The fix is not a cleverer mapping -- it is to
**coarsen BOTH sides to the same taxonomy** (`to_common()` on the GT, on Qwen-Drive's output,
and on BEVFusion's output alike) so the coarsening bias applies equally and cancels. Any
number produced by scoring one side in its native taxonomy against the other in this one is
not a headroom measurement; it is an artefact.

`--report-collapse` prints, per frame, what fraction of occupied GT voxels lost information
in the collapse, so the size of the concession is on the record rather than in a footnote.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- taxonomies

OCC3D_CLASSES = (
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
    'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface', 'other_flat',
    'sidewalk', 'terrain', 'manmade', 'vegetation', 'free')
OCC3D_FREE = 17

# configuration_perception.OCC_CLASS_NAMES
QD_CLASSES = ('vehicle', 'czone_sign', 'bicycle', 'generic_object', 'pedestrian',
              'traffic_cone', 'barrier', 'driveable', 'background', 'empty')
QD_EMPTY = 9
QD_CZONE_SIGN = 1                      # unreachable from nuScenes GT -- see the docstring

OCC3D_TO_QD = {
    'others': 'generic_object',
    'barrier': 'barrier',
    'bicycle': 'bicycle',
    'bus': 'vehicle',
    'car': 'vehicle',
    'construction_vehicle': 'vehicle',
    'motorcycle': 'bicycle',           # LOSSY: no two-wheeled-motor class exists
    'pedestrian': 'pedestrian',
    'traffic_cone': 'traffic_cone',
    'trailer': 'vehicle',
    'truck': 'vehicle',
    'driveable_surface': 'driveable',
    'other_flat': 'background',
    'sidewalk': 'background',
    'terrain': 'background',
    'manmade': 'background',
    'vegetation': 'background',
    'free': 'empty',
}

#: nuScenes detection classes -> the same 7-class taxonomy, for scoring BEVFusion on it.
NUSC_DET_TO_QD = {
    'car': 'vehicle', 'truck': 'vehicle', 'bus': 'vehicle', 'trailer': 'vehicle',
    'construction_vehicle': 'vehicle',
    'bicycle': 'bicycle', 'motorcycle': 'bicycle',
    'pedestrian': 'pedestrian', 'traffic_cone': 'traffic_cone', 'barrier': 'barrier',
}

#: Classes that lose information in the collapse (>1 source class maps onto them).
_MERGED = ('vehicle', 'bicycle', 'background')


def _build_lut() -> np.ndarray:
    lut = np.full(len(OCC3D_CLASSES), QD_EMPTY, np.uint8)
    for i, name in enumerate(OCC3D_CLASSES):
        lut[i] = QD_CLASSES.index(OCC3D_TO_QD[name])
    return lut


OCC3D_LUT = _build_lut()


def occ3d_to_qd(occ: np.ndarray) -> np.ndarray:
    """Map an Occ3D-nuScenes semantic grid into Qwen-Drive's 10-class space."""
    if occ.max() >= len(OCC3D_CLASSES):
        raise ValueError(f'label {int(occ.max())} out of range for Occ3D (18 classes)')
    return OCC3D_LUT[occ]


def to_common(labels: np.ndarray, taxonomy: str) -> np.ndarray:
    """Coarsen either side into the common 10-class space.

    ``taxonomy='occ3d'`` maps 18 -> 10; ``'qd'`` is already common and returned unchanged
    (present so call sites read symmetrically and no side is silently left un-coarsened).
    """
    if taxonomy == 'occ3d':
        return occ3d_to_qd(labels)
    if taxonomy == 'qd':
        return labels
    raise ValueError(f'unknown taxonomy {taxonomy!r}')


def collapse_report(occ3d: np.ndarray) -> dict:
    """How much information the collapse destroys on THIS frame.

    ``merged_frac`` is the fraction of *occupied* GT voxels whose class is one of the merged
    targets, i.e. voxels for which the common taxonomy can no longer distinguish the mistake
    a model might make. Quote it next to any headroom number.
    """
    occ = np.asarray(occ3d)
    occupied = occ != OCC3D_FREE
    n = int(occupied.sum())
    if n == 0:
        return dict(n_occupied=0, merged_frac=float('nan'), per_target={})
    mapped = occ3d_to_qd(occ)[occupied]
    per = {}
    for t in _MERGED:
        per[t] = float((mapped == QD_CLASSES.index(t)).sum()) / n
    return dict(n_occupied=n, merged_frac=float(sum(per.values())), per_target=per)


# ---------------------------------------------------------------- grid check

def assert_grid_compatible(qd_config: dict) -> None:
    """Fail loudly if Qwen-Drive's occ grid ever stops being voxel-identical to Occ3D.

    Reads the released `perception/config.json` dict. The whole comparison rests on this,
    and it is one `==` -- so it is asserted, not assumed.
    """
    pc = list(qd_config['nuscenes_occ_pc_range'])
    vs = list(qd_config['nuscenes_occ_voxel_size'])
    ph = int(qd_config['occ_pillar_h'])
    exp_pc = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
    if pc != exp_pc:
        raise AssertionError(f'occ pc_range {pc} != Occ3D {exp_pc}')
    nx = round((pc[3] - pc[0]) / vs[0])
    ny = round((pc[4] - pc[1]) / vs[1])
    dz = (pc[5] - pc[2]) / ph
    if (nx, ny, ph) != (200, 200, 16) or abs(dz - 0.4) > 1e-9:
        raise AssertionError(f'grid {(nx, ny, ph)} dz={dz} != Occ3D (200,200,16) dz=0.4')


# ---------------------------------------------------------------- self-test

def _self_test() -> int:
    ok = True

    def chk(name, cond):
        nonlocal ok
        print(f'  {"PASS" if cond else "FAIL"}  {name}')
        ok &= bool(cond)

    chk('every Occ3D class has a target', len(OCC3D_TO_QD) == len(OCC3D_CLASSES))
    chk('free -> empty', OCC3D_LUT[OCC3D_FREE] == QD_EMPTY)
    chk('czone_sign unreachable', QD_CLASSES.index('czone_sign') not in set(OCC3D_LUT))
    chk('5 nuScenes vehicle classes collapse to 1',
        sum(v == 'vehicle' for v in OCC3D_TO_QD.values()) == 5)
    chk('5 background classes collapse to 1',
        sum(v == 'background' for v in OCC3D_TO_QD.values()) == 5)
    chk('LUT dtype/len', OCC3D_LUT.shape == (18,) and OCC3D_LUT.dtype == np.uint8)

    g = np.full((200, 200, 16), OCC3D_FREE, np.uint8)
    g[0, 0, 0] = OCC3D_CLASSES.index('car')
    g[0, 0, 1] = OCC3D_CLASSES.index('truck')
    g[0, 0, 2] = OCC3D_CLASSES.index('pedestrian')
    m = occ3d_to_qd(g)
    chk('car and truck become the same label',
        m[0, 0, 0] == m[0, 0, 1] == QD_CLASSES.index('vehicle'))
    chk('pedestrian survives', m[0, 0, 2] == QD_CLASSES.index('pedestrian'))
    chk('free voxels stay empty', (m[1:] == QD_EMPTY).all())
    r = collapse_report(g)
    chk('collapse_report counts occupied only', r['n_occupied'] == 3)
    chk('collapse_report merged_frac = 2/3', abs(r['merged_frac'] - 2 / 3) < 1e-9)

    try:
        assert_grid_compatible(dict(nuscenes_occ_pc_range=[-40, -40, -1, 40, 40, 5.4],
                                    nuscenes_occ_voxel_size=[0.4, 0.4, 6.4],
                                    occ_pillar_h=16))
        chk('released grid is Occ3D-compatible', True)
    except AssertionError as e:
        chk(f'released grid is Occ3D-compatible ({e})', False)
    try:
        assert_grid_compatible(dict(nuscenes_occ_pc_range=[-50, -50, -4, 50, 50, 4],
                                    nuscenes_occ_voxel_size=[0.5, 0.5, 0.5],
                                    occ_pillar_h=16))
        chk('a WRONG grid is rejected', False)
    except AssertionError:
        chk('a WRONG grid is rejected', True)

    print('\n' + ('ALL PASS' if ok else 'FAILURES ABOVE'))
    return 0 if ok else 1


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(_self_test())
    print(__doc__)
    print('\nOcc3D -> Qwen-Drive:')
    for i, n in enumerate(OCC3D_CLASSES):
        print(f'  {i:2d} {n:22s} -> {OCC3D_LUT[i]:2d} {QD_CLASSES[OCC3D_LUT[i]]}')
