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
POINTS = {
    2000: {"finetune": (0.0637, 0.263), "scratch": (0.0409, 0.129)},
    4000: {"finetune": (0.0994, 0.268), "scratch": (0.0475, 0.084)},
    8000: {"finetune": (0.1590, 0.371), "scratch": (0.1142, 0.299)},
}


def main():
    ap = argparse.ArgumentParser(description="Plot the occ->det label-efficiency curve.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    budgets = sorted(b for b in POINTS if POINTS[b]["finetune"][0] is not None)
    ft_map = [POINTS[b]["finetune"][0] for b in budgets]
    sc_map = [POINTS[b]["scratch"][0] for b in budgets]
    ft_ped = [POINTS[b]["finetune"][1] for b in budgets]
    sc_ped = [POINTS[b]["scratch"][1] for b in budgets]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    a1.plot(budgets, ft_map, "o-", color="#1f77b4", lw=2, label="occ-pretrained (finetune)")
    a1.plot(budgets, sc_map, "s--", color="#d62728", lw=2, label="from-scratch")
    for b in budgets:
        r = POINTS[b]["finetune"][0] / max(POINTS[b]["scratch"][0], 1e-6)
        a1.annotate(f"{r:.1f}x", (b, POINTS[b]["finetune"][0]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, color="#1f77b4")
    a1.set_xlabel("detection label budget (frames)"); a1.set_ylabel("official nuScenes mAP")
    a1.set_title("Label efficiency: occupancy pretraining vs from-scratch")
    a1.legend(); a1.grid(alpha=0.3)

    a2.plot(budgets, ft_ped, "o-", color="#1f77b4", lw=2, label="occ-pretrained")
    a2.plot(budgets, sc_ped, "s--", color="#d62728", lw=2, label="from-scratch")
    a2.set_xlabel("detection label budget (frames)"); a2.set_ylabel("pedestrian AP")
    a2.set_title("The gain is rare-object transfer (pedestrian)")
    a2.legend(); a2.grid(alpha=0.3)

    fig.suptitle("Occupancy->detection transfer is label-efficient (occ pretraining, nuScenes)",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[plot] wrote {args.out}")


if __name__ == "__main__":
    main()
