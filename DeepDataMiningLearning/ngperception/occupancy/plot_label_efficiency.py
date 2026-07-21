"""
ngperception.occupancy.plot_label_efficiency
=============================================
Plot the occupancy->detection **label-efficiency curve** (the positive result): official nuScenes
mAP vs detection label budget, occ-pretrained (finetune) vs from-scratch, plus the rare-class
(pedestrian) sub-curve that carries the transfer benefit. Numbers are the official
`eval_det_ablation_official.py` mAP over the 6019-frame val set.

Edit POINTS below with the measured (mAP, pedestrian AP) per (budget, arm), then:
    python -m DeepDataMiningLearning.ngperception.occupancy.plot_label_efficiency \
        --out DeepDataMiningLearning/ngperception/docs/label_efficiency_curve.png
"""
from __future__ import annotations
import argparse

# budget -> {arm: (mAP, pedestrian_AP)} — official nuScenes val (center-distance)
# occ3d = Occ3D-GT-pretrained (expensive labels); render2d = label-free pretext; scratch = floor
POINTS = {
    2000: {"occ3d": (0.0637, 0.263), "render2d": (0.0403, 0.142), "scratch": (0.0409, 0.129)},
    4000: {"occ3d": (0.0994, 0.268), "render2d": (0.0651, 0.181), "scratch": (0.0475, 0.084)},
    8000: {"occ3d": (0.1590, 0.371), "render2d": (0.1150, 0.305), "scratch": (0.1142, 0.299)},
}
ARMS = [("occ3d", "occ pretrain (Occ3D-GT, expensive)", "#1f77b4", "o-"),
        ("render2d", "occ pretrain (render2d, LABEL-FREE)", "#2ca02c", "^-"),
        ("scratch", "from-scratch", "#d62728", "s--")]


def main():
    ap = argparse.ArgumentParser(description="Plot the occ->det label-efficiency curve.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default=None, help="results.csv from run_label_efficiency.sh (seeded)")
    args = ap.parse_args()
    import csv as _csv
    import statistics as _st
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # seeded arms from CSV -> {arm: {budget: {'mAP':[...], 'ped':[...]}}}
    seeded = {}
    if args.csv:
        with open(args.csv) as f:
            for row in _csv.DictReader(f):
                try:
                    m, p = float(row["mAP"]), float(row["ped_AP"])
                except (ValueError, KeyError):
                    continue
                d = seeded.setdefault(row["arm"], {}).setdefault(int(row["budget"]), {"mAP": [], "ped": []})
                d["mAP"].append(m); d["ped"].append(p)

    def mean_std(vals):
        return (_st.mean(vals), (_st.pstdev(vals) if len(vals) > 1 else 0.0))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    # reference (single-seed) upper bound: occ3d-GT (expensive labels)
    rb = sorted(POINTS)
    a1.plot(rb, [POINTS[b]["occ3d"][0] for b in rb], "o:", color="#1f77b4", lw=1.5, alpha=0.7,
            label="occ pretrain (Occ3D-GT, expensive) — ref")
    a2.plot(rb, [POINTS[b]["occ3d"][1] for b in rb], "o:", color="#1f77b4", lw=1.5, alpha=0.7,
            label="occ pretrain (Occ3D-GT) — ref")
    # seeded arms with error bars
    STYLE = {"voxelsoft": ("occ pretrain (voxel-soft, LABEL-FREE)", "#2ca02c", "^-"),
             "scratch": ("from-scratch", "#d62728", "s--")}
    for arm, (label, color, style) in STYLE.items():
        if arm not in seeded:
            continue
        bs = sorted(seeded[arm])
        for ax, key in ((a1, "mAP"), (a2, "ped")):
            ms = [mean_std(seeded[arm][b][key]) for b in bs]
            n = len(seeded[arm][bs[0]][key])
            ax.errorbar(bs, [m for m, _ in ms], yerr=[s for _, s in ms], fmt=style, color=color,
                        lw=2, capsize=4, label=f"{label} (n={n})")
    a1.set_xlabel("detection label budget (frames)"); a1.set_ylabel("official nuScenes mAP")
    a1.set_title("Label efficiency (official mAP)"); a1.legend(fontsize=8); a1.grid(alpha=0.3)
    a2.set_xlabel("detection label budget (frames)"); a2.set_ylabel("pedestrian AP")
    a2.set_title("Rare-object transfer (pedestrian)"); a2.legend(fontsize=8); a2.grid(alpha=0.3)
    fig.suptitle("Occupancy->detection: does a LABEL-FREE occ pretext give label-efficient detection?",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[plot] wrote {args.out}  (seeded arms: {sorted(seeded)})")


if __name__ == "__main__":
    main()
