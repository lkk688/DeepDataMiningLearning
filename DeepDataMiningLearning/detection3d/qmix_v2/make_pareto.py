"""Source-retention vs target-gain tradeoff as the nuScenes mix fraction
varies (B14 25% / B30 12.5% / B31 6.25%). x = nuScenes NDS (source
retention, full val), y = Waymo Macro AP@2m (target transfer, 300-frame
stride-20). All three share the same eval protocol. CPU-only, hardcoded
from the completed runs."""
from __future__ import annotations
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# (label, nuScenes NDS [full val], Waymo Macro AP@2m [300-frame], mix%)
PTS = [
    ('B14',  0.482, 0.258, 25.0),
    ('B30',  0.419, 0.230, 12.5),
    ('B31',  0.315, 0.224, 6.25),
]

fig, ax = plt.subplots(figsize=(5.2, 4.0))
xs = [p[1] for p in PTS]; ys = [p[2] for p in PTS]
ax.plot(xs, ys, '-', color='0.6', lw=1.2, zorder=1)
sizes = [p[3] * 14 for p in PTS]
sc = ax.scatter(xs, ys, s=sizes, c=[p[3] for p in PTS], cmap='viridis',
                edgecolors='k', linewidths=0.8, zorder=3)
for lbl, nds, mac, frac in PTS:
    ax.annotate(f'{lbl}\nnuScenes {frac:g}%', (nds, mac),
                textcoords='offset points', xytext=(8, -4), fontsize=8)

ax.set_xlabel('nuScenes NDS  (source retention, full val)')
ax.set_ylabel('Waymo Macro AP@2m  (target transfer, 300-frame)')
ax.set_title('Mixed-source ratio: source anchor is complementary,\n'
             'not competing — 25% dominates on both axes')
ax.grid(True, ls=':', alpha=0.5)
cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
cb.set_label('nuScenes mix fraction (%)', fontsize=8)
# arrow showing the "more nuScenes" direction
ax.annotate('more nuScenes anchor', xy=(0.46, 0.252), xytext=(0.34, 0.247),
            fontsize=7.5, color='0.35',
            arrowprops=dict(arrowstyle='->', color='0.5', lw=1.0))

out = os.path.join(HERE, 'pareto_mix_ratio.pdf')
fig.savefig(out, bbox_inches='tight')
print(f'wrote {out}')
plt.close(fig)
