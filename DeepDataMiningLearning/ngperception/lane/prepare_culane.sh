#!/usr/bin/env bash
# ============================================================================
# prepare_culane.sh — download + extract CULane into the layout our loader wants
# ============================================================================
# CULane (https://xingangpan.github.io/projects/CULane.html) ships as tarballs on
# Google Drive:
#   https://drive.google.com/drive/folders/1mSLgwVTiaUMAb4AVOWwlCD5JcWdrwpvu
#
# For our pure-torch CLRNet (clrnet.py / train_clrnet.py) we only need:
#   • driver_*_*frame.tar.gz  (6 tars) — the .jpg images AND the .lines.txt lane
#                                        annotations live together in these
#   • list.tar.gz             — the train/val/test split lists (+ train_gt.txt)
# NOT needed (only for segmentation-based methods / aux-seg): laneseg_label_w16*,
# annotations_new (a cleaned re-annotation), video_example.
#
# Usage:
#   bash prepare_culane.sh download   # gdown the whole Drive folder to $ROOT
#   bash prepare_culane.sh extract    # untar the driver tars + list into place
#   bash prepare_culane.sh verify     # sanity-check the resulting layout
#   bash prepare_culane.sh all        # download → extract → verify
#
# Final layout ($ROOT):
#   driver_23_30frame/…/*.jpg + *.lines.txt   (and the other 5 driver dirs)
#   list/{train,val,test}.txt  list/{train,val}_gt.txt  list/test_split/*.txt
# ============================================================================
set -euo pipefail

ROOT="${CULANE_ROOT:-/mnt/e/Shared/Dataset/CULane}"
FOLDER="https://drive.google.com/drive/folders/1mSLgwVTiaUMAb4AVOWwlCD5JcWdrwpvu"
PY="${PY:-/home/lkk688/miniconda/envs/py312/bin/python}"
GDOWN="${GDOWN:-/home/lkk688/miniconda/envs/py312/bin/gdown}"
DRIVERS=(driver_23_30frame driver_37_30frame driver_100_30frame \
         driver_161_90frame driver_182_30frame driver_193_90frame)

download() {
  mkdir -p "$ROOT"
  echo "[culane] downloading Drive folder -> $ROOT (this is ~100 GB incl. optional seg labels)"
  "$GDOWN" --folder "$FOLDER" -O "$ROOT"
}

extract() {
  cd "$ROOT"
  for d in "${DRIVERS[@]}"; do
    if [ -d "$ROOT/$d" ]; then echo "[culane] $d already extracted"; continue; fi
    if [ -f "$ROOT/$d.tar.gz" ]; then
      echo "[culane] extracting $d.tar.gz …"; tar -xzf "$d.tar.gz" -C "$ROOT"
    else echo "[culane] WARN: $d.tar.gz missing (download incomplete?)"; fi
  done
  if [ -f "$ROOT/list.tar.gz" ] && [ ! -d "$ROOT/list" ]; then
    echo "[culane] extracting list.tar.gz …"; tar -xzf "$ROOT/list.tar.gz" -C "$ROOT"
  fi
  echo "[culane] extract done"
}

verify() {
  echo "[culane] verifying $ROOT"
  local ok=1
  for d in "${DRIVERS[@]}"; do
    [ -d "$ROOT/$d" ] && echo "  ok  $d/" || { echo "  MISSING $d/"; ok=0; }
  done
  for f in list/train_gt.txt list/val_gt.txt list/test.txt; do
    [ -f "$ROOT/$f" ] && echo "  ok  $f ($(wc -l < "$ROOT/$f") lines)" || { echo "  MISSING $f"; ok=0; }
  done
  # a .jpg should have a sibling .lines.txt
  local jpg; jpg=$(find "$ROOT/${DRIVERS[0]}" -name '*.jpg' 2>/dev/null | head -1 || true)
  if [ -n "$jpg" ]; then
    [ -f "${jpg%.jpg}.lines.txt" ] && echo "  ok  sample .lines.txt present" \
      || echo "  NOTE: ${jpg%.jpg}.lines.txt absent (some frames are unlabeled — normal)"
  fi
  [ "$ok" = 1 ] && echo "[culane] VERIFY OK — ready for train_clrnet --dataset culane --root $ROOT" \
                || echo "[culane] VERIFY INCOMPLETE — re-run download/extract"
}

case "${1:-all}" in
  download) download ;;
  extract)  extract ;;
  verify)   verify ;;
  all)      download; extract; verify ;;
  *) echo "usage: $0 {download|extract|verify|all}"; exit 1 ;;
esac
