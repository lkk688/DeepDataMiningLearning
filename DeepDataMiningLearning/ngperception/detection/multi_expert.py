"""
ngperception.detection.multi_expert
====================================

**Path B — modality-routed multi-expert 3-D detection.** The single-model distillation (Path A,
det TUTORIAL §15) proved you cannot fix camera-only detection inside our BEV model. Path B instead
**routes each modality to its best-available expert** — every modality a *strong* detector:

    camera-only  -> PETR            (mmdet3d, camera,        mAP 0.383 / NDS 0.391)
    lidar-only   -> BEVFusion-L     (mmdet3d, LiDAR,         mAP 0.643 / NDS 0.691)
    cam+lidar    -> BEVFusion-LC    (mmdet3d, fusion,        mAP 0.684 / NDS 0.712)
    occupancy    -> ours (LSSOccupancy, any modality; occ mIoU 0.302 cam / 0.558 fusion)
    (all three detection experts reproduced on the full nuScenes val split, py310 + spconv 2.3.6)

Each mmdet3d expert (PETR / BEVFusion) registers conflicting modules, so they run in **isolated
subprocesses** (this dispatcher shells to `tools/test.py`). The occupancy expert is ours, native.
This mirrors `worldmodel_drive/scripts/run_pipeline.py`, which also late-fuses the experts.

    # evaluate the expert for a given modality on nuScenes val:
    python -m DeepDataMiningLearning.ngperception.detection.multi_expert --modality camera --eval
    python -m DeepDataMiningLearning.ngperception.detection.multi_expert --modality fused  --eval
    # print the routing policy + combined results table:
    python -m DeepDataMiningLearning.ngperception.detection.multi_expert --table
"""
from __future__ import annotations
import argparse
import subprocess

MMDET3D = "/data/rnd-liu/MyRepo/mmdetection3d"

# modality -> best expert (mmdet3d config/ckpt relative to MMDET3D; metrics are official nuScenes val)
EXPERTS = {
    "camera": dict(name="PETR", mAP=0.383, NDS=0.391,
                   config="projects/PETR/configs/petr_vovnet_gridmask_p4_800x320.py",
                   ckpt="modelzoo_mmdetection3d/petr_vovnet_gridmask_p4_800x320.pth"),
    "lidar": dict(name="BEVFusion-L", mAP=0.643, NDS=0.691,   # reproduced here on nuScenes val
                  config="projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
                  ckpt="modelzoo_mmdetection3d/bevfusion_lidar_spconv236.pth"),
    "fused": dict(name="BEVFusion-LC", mAP=0.684, NDS=0.712,  # reproduced here on nuScenes val
                  config="projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
                  ckpt="modelzoo_mmdetection3d/bevfusion_lidarcam_spconv236.pth"),
}
# occupancy is ours (native, any modality) — LSSOccupancy; det for our BEV model (context/ablation)
OURS = dict(occ_mIoU_camera=0.302, occ_mIoU_fusion=0.558, det_mAP_fusion=0.391)


def print_table():
    print("\n=== Path B — modality-routed multi-expert (official nuScenes val) ===")
    print(f"{'modality':<12}{'expert':<16}{'det mAP':<9}{'NDS':<7}  occupancy (ours)")
    print("-" * 66)
    for mod, e in EXPERTS.items():
        occ = (f"mIoU {OURS['occ_mIoU_camera']}" if mod == "camera" else
               f"mIoU {OURS['occ_mIoU_fusion']}")
        print(f"{mod:<12}{e['name']:<16}{e['mAP']:<9}{e['NDS']:<7}  {occ}")
    print("-" * 66)
    print("Contrast: single-model camera det (Path A distill) ~0.008 -> PETR routing gives 0.383.")
    print("Our BEV occ-backbone detector (fused, pure-PyTorch) reaches det mAP "
          f"{OURS['det_mAP_fusion']} — the strongest *native* line; BEVFusion (spconv) is the SOTA expert.")


def run_expert(modality, indices=None):
    if modality not in EXPERTS:
        raise SystemExit(f"no mmdet3d expert for modality '{modality}' (choices: {list(EXPERTS)})")
    e = EXPERTS[modality]
    cmd = ["python", "tools/test.py", e["config"], e["ckpt"], "--work-dir", f"work_dirs/expert_{modality}"]
    if indices:
        cmd += ["--cfg-options", f"test_dataloader.dataset.indices={indices}",
                "test_evaluator.ann_file=data/nuscenes/nuscenes_infos_val.pkl"]
    print(f"[multi-expert] {modality} -> {e['name']} (expected mAP {e['mAP']})\n  cd {MMDET3D} && {' '.join(cmd)}")
    subprocess.run(cmd, cwd=MMDET3D, check=True)


def main():
    ap = argparse.ArgumentParser(description="Path B — modality-routed multi-expert detection.")
    ap.add_argument("--modality", choices=["camera", "lidar", "fused"], default=None)
    ap.add_argument("--eval", action="store_true", help="run the routed expert on nuScenes val")
    ap.add_argument("--indices", type=int, default=None, help="subset N val frames (quick check)")
    ap.add_argument("--table", action="store_true", help="print the routing policy + combined table")
    args = ap.parse_args()
    if args.table or not args.modality:
        print_table()
    if args.modality and args.eval:
        run_expert(args.modality, args.indices)


if __name__ == "__main__":
    main()
