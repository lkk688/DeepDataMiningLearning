#!/usr/bin/env bash
# Label-efficiency: the LABEL-BASED arm (Occ3D-GT-supervised occ pretext, lss_occ_full) — the decisive
# comparison vs the label-free voxel-soft arm (run_label_efficiency.sh) and scratch. Same seeded setup,
# separate CSV (results_occ3d.csv) so it can run in parallel with the label-free grid (which is
# dataloader-bound, leaving GPU compute idle). Answers: does label-BASED occ pretraining give the
# label-efficiency gain that the label-free pretext does not?
set -e
cd /fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
export PYTHONPATH=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning
source ~/.bashrc 2>/dev/null || true; conda activate py310 2>/dev/null || true
ROOT=/data/rnd-liu/Datasets/nuScenes
NUSC=$ROOT/v1.0-trainval; GTS=$ROOT/v1.0-trainval/gts
OUT=DeepDataMiningLearning/ngperception/output/label_eff
PRETRAIN=DeepDataMiningLearning/ngperception/output/lss_occ_full/lss_occ.pth
CSV=$OUT/results_occ3d.csv
PY="python -m DeepDataMiningLearning.ngperception.occupancy"
MCFG="--backbone dinov2_base --decoder-layers 4 --decoder-hidden 96 --refine-iters 1 --det-head center"
mkdir -p $OUT
[ -f $CSV ] || echo "arm,budget,seed,mAP,NDS,ped_AP" > $CSV

for seed in 1 2 3; do
  for budget in 2000 4000 8000; do
    tag=occ3d_b${budget}_s${seed}
    grep -q "^occ3d,${budget},${seed}," $CSV && { echo "[le-occ3d] skip $tag"; continue; }
    echo "===== [le-occ3d] train $tag ====="
    $PY.train_det_ablation --nusc $NUSC --gts $GTS --pretrained $PRETRAIN $MCFG \
        --max-samples $budget --val-samples 200 --epochs 12 --batch-size 8 --lr 2e-3 --cosine \
        --num-workers 8 --seed $seed --out-dir $OUT/$tag
    $PY.eval_det_ablation_official --nusc $NUSC --gts $GTS \
        --ckpt $OUT/$tag/det_abl.pth --out-dir ${OUT}/${tag}_eval > ${OUT}/${tag}_eval.log 2>&1 || true
    mAP=$(grep -oE "mAP = [0-9.]+" ${OUT}/${tag}_eval.log | tail -1 | grep -oE "[0-9.]+")
    NDS=$(grep -oE "NDS = [0-9.]+" ${OUT}/${tag}_eval.log | tail -1 | grep -oE "[0-9.]+")
    ped=$(grep -E "pedestrian" ${OUT}/${tag}_eval.log | tail -1 | grep -oE "[0-9.]+" | tail -1)
    echo "occ3d,${budget},${seed},${mAP:-NA},${NDS:-NA},${ped:-NA}" >> $CSV
    echo "[le-occ3d] $tag -> mAP=${mAP} NDS=${NDS} ped=${ped}"
    rm -f $OUT/$tag/det_abl.pth
  done
done
echo "[le-occ3d] DONE -> $CSV"
