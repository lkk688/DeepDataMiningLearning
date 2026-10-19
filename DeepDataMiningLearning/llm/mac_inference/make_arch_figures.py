"""Vector (SVG) architecture figures for TUTORIAL_mac_inference.md.

python mac_inference/make_arch_figures.py      # -> docs/mac_inference_figs/*.svg

Explicit colours on a white background, so the figures stay legible in GitHub's light and dark themes.
"""
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent.parent / "docs/mac_inference_figs"
INK, MUTED, LINE, BOX = "#1f2328", "#57606a", "#8c959f", "#f6f8fa"
RAM, RAM_BG = "#2a78d6", "#e8f1fc"          # resident in memory / selected
SSD, SSD_BG = "#d4561f", "#fdeee7"          # on SSD / recomputed
STATE, STATE_BG = "#13875b", "#e5f5ee"      # fixed-size recurrent state / reused


class SVG:
    def __init__(self, w, h, title):
        self.w, self.h, self.parts = w, h, []
        self.parts.append(f'<rect width="{w}" height="{h}" fill="#ffffff"/>')
        self.text(20, 30, title, size=16, weight="600")

    def text(self, x, y, s, size=13, color=INK, anchor="start", weight="400", italic=False):
        style = ' font-style="italic"' if italic else ""
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
                          f'font-weight="{weight}"{style}>{escape(s)}</text>')

    def box(self, x, y, w, h, lines=(), stroke=LINE, fill=BOX, size=13, rx=8, dash=False, bold_first=False, color=INK):
        d = ' stroke-dasharray="5 4"' if dash else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="1.5"{d}/>')
        n = len(lines)
        for i, s in enumerate(lines):
            yy = y + h / 2 + (i - (n - 1) / 2) * (size + 4) + size * 0.36
            self.text(x + w / 2, yy, s, size=size if i else size, color=color if i == 0 or not bold_first else MUTED,
                      anchor="middle", weight="600" if (bold_first and i == 0) else "400")

    def arrow(self, x1, y1, x2, y2, label=None, color=MUTED, dash=False, lx=None, ly=None, anchor="middle", size=12):
        d = ' stroke-dasharray="5 4"' if dash else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.6"'
                          f'{d} marker-end="url(#a)"/>')
        if label:
            self.text(lx if lx is not None else (x1 + x2) / 2, ly if ly is not None else (y1 + y2) / 2 - 6,
                      label, size=size, color=MUTED, anchor=anchor)

    def line(self, x1, y1, x2, y2, color=LINE, width=1.5, dash=False):
        d = ' stroke-dasharray="5 4"' if dash else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{d}/>')

    def path(self, d, color=MUTED, width=1.6, arrow=True, dash=False):
        dd = ' stroke-dasharray="5 4"' if dash else ""
        m = ' marker-end="url(#a)"' if arrow else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{dd}{m}/>')

    def save(self, name, label):
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
               f'height="{self.h}" role="img" aria-label="{escape(label)}" '
               f'font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif">'
               f'<title>{escape(label)}</title>'
               f'<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
               f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs>'
               + "".join(self.parts) + "</svg>\n")
        (OUT / name).write_text(svg)


def arch_27b():
    s = SVG(860, 610, "Qwen3.8-27B: a dense hybrid, 64 layers, one residual stream")
    s.box(40, 60, 170, 40, ["text tokens"])
    s.box(40, 120, 170, 40, ["image patches"])
    s.box(260, 60, 250, 40, ["token embedding", ], size=13)
    s.text(385, 116, "248,320 × 5,120", size=12, color=MUTED, anchor="middle")
    s.box(260, 130, 250, 40, ["vision encoder: 27 layers, 0.9 GB"])
    s.arrow(210, 80, 258, 80)
    s.arrow(210, 140, 258, 150)
    s.path("M385,170 L385,206", color=MUTED)
    s.path("M510,80 C560,80 560,190 420,206", color=MUTED)
    s.text(570, 140, "image features join the", size=12, color=MUTED)
    s.text(570, 156, "token sequence", size=12, color=MUTED)
    # the 16 x 4-layer group
    s.box(200, 210, 370, 240, [], stroke=INK, fill="#ffffff", rx=12)
    s.text(216, 234, "× 16 groups = 64 layers", size=13, weight="600")
    s.box(225, 250, 320, 70, ["Gated DeltaNet (linear attention)", "→ dense SwiGLU MLP, 17,408"], stroke=STATE, fill=STATE_BG)
    s.text(535, 290, "× 3", size=14, weight="600", color=STATE, anchor="end")
    s.arrow(385, 320, 385, 352)
    s.box(225, 355, 320, 70, ["gated full attention: 24 Q / 4 KV heads, d 256", "→ dense SwiGLU MLP, 17,408"], stroke=SSD, fill=SSD_BG)
    s.text(535, 395, "× 1", size=14, weight="600", color=SSD, anchor="end")
    # what each layer type keeps between tokens
    s.line(545, 285, 600, 285, color=STATE, dash=True)
    s.text(606, 280, "keeps a fixed-size state:", size=12, color=STATE, weight="600")
    s.text(606, 296, "48 heads × 128 × 128, fp32", size=12, color=STATE)
    s.text(606, 312, "0.154 GB for all 48 layers", size=12, color=STATE)
    s.line(545, 390, 600, 390, color=SSD, dash=True)
    s.text(606, 385, "keeps a KV cache that grows:", size=12, color=SSD, weight="600")
    s.text(606, 401, "16 layers × 2 × 4 × 256 × 2 B", size=12, color=SSD)
    s.text(606, 417, "= 64 KiB per token", size=12, color=SSD)
    # output heads
    s.arrow(385, 450, 385, 488, label="h_t", lx=395, ly=474, anchor="start")
    s.box(260, 490, 250, 40, ["RMSNorm → LM head 248,320"])
    s.arrow(385, 530, 385, 556)
    s.box(285, 558, 200, 34, ["next token x_t+1"], size=13)
    s.box(590, 490, 230, 40, ["MTP head: 1 attention layer"], stroke=RAM, fill=RAM_BG)
    s.path("M420,450 C470,470 540,470 588,500", color=MUTED)
    s.path("M485,575 C540,575 560,545 600,532", color=MUTED)
    s.text(552, 476, "h_t", size=12, color=MUTED)
    s.text(500, 592, "x_t+1", size=12, color=MUTED)
    s.arrow(705, 530, 705, 556)
    s.box(605, 558, 200, 34, ["draft token x̂_t+2"], stroke=RAM, fill=RAM_BG)
    s.save("arch_qwen38_27b.svg", "Qwen3.8-27B: 16 groups of three Gated DeltaNet layers and one full-attention layer, "
                                   "each followed by a dense MLP; an MTP head drafts the token after next.")


def arch_flashnext():
    s = SVG(980, 720, "Qwen3.8-Flash-Next: 125B MoE (6B active), 4 residual streams, n-gram memory")
    s.box(40, 60, 170, 40, ["text tokens"])
    s.box(260, 60, 250, 40, ["token embedding 248,320 × 2,560"], stroke=RAM, fill=RAM_BG)
    s.arrow(210, 80, 258, 80)
    s.box(560, 60, 380, 56, ["n-gram embedding: hashed bigrams + trigrams", "29.8 GiB table, 16 rows of 160 per token"],
          stroke=SSD, fill=SSD_BG)
    s.path("M125,100 C125,160 600,130 640,116", color=MUTED, dash=True)
    s.text(430, 146, "last 3 tokens → n-gram hash", size=12, color=MUTED)
    s.arrow(385, 100, 385, 182, label="copy into 4 streams", lx=375, ly=170, anchor="end")
    # stack
    s.box(170, 185, 560, 330, [], stroke=INK, fill="#ffffff", rx=12)
    s.text(186, 209, "× 12 groups = 48 layers", size=13, weight="600")
    for i, x in enumerate((196, 206, 216, 226)):
        s.line(x, 225, x, 500, color=RAM, width=2.2)
    s.text(190, 512, "4 streams × 2,560", size=11, color=RAM)
    s.box(250, 225, 440, 92, [], stroke=STATE, fill=STATE_BG)
    s.text(470, 247, "Gated DeltaNet → MoE", size=13, anchor="middle", weight="600")
    s.text(470, 267, "each sublayer reads and writes the 4 streams", size=12, anchor="middle", color=MUTED)
    s.text(470, 283, "through a gated residual", size=12, anchor="middle", color=MUTED)
    s.text(470, 303, "layer 2 adds the n-gram memory", size=12, anchor="middle", color=SSD)
    s.text(700, 275, "× 3", size=14, weight="600", color=STATE)
    s.path("M750,116 C790,160 760,280 692,295", color=SSD, dash=True)
    s.text(772, 210, "gated add at layer 2", size=12, color=SSD)
    s.arrow(470, 317, 470, 345)
    s.box(250, 347, 440, 76, [], stroke=SSD, fill=SSD_BG)
    s.text(470, 371, "Qwen Sparse Attention → MoE", size=13, anchor="middle", weight="600")
    s.text(470, 391, "attends to ≤ 2,048 selected tokens (512 blocks of 4)", size=12, anchor="middle", color=MUTED)
    s.text(470, 409, "24 Q / 2 KV heads, d 256; KV cache 24 KiB per token", size=12, anchor="middle", color=MUTED)
    s.text(700, 390, "× 1", size=14, weight="600", color=SSD)
    # MoE detail
    s.box(250, 440, 440, 60, [], stroke=LINE, fill=BOX)
    s.text(470, 462, "every MoE: router picks 10 of 512 experts (+1 shared)", size=12.5, anchor="middle")
    s.text(470, 482, "expert = SwiGLU 2,560 → 640 → 2,560, 2.8 MB at 4-bit", size=12, anchor="middle", color=MUTED)
    s.box(760, 405, 190, 110, [], stroke=SSD, fill=SSD_BG)
    s.text(855, 428, "512 experts × 48 layers", size=12, anchor="middle", weight="600", color=SSD)
    s.text(855, 446, "63.3 GiB", size=12, anchor="middle", color=SSD)
    s.text(855, 464, "a fraction stays in RAM,", size=12, anchor="middle", color=SSD)
    s.text(855, 480, "the rest is read from SSD", size=12, anchor="middle", color=SSD)
    s.text(855, 496, "when routed", size=12, anchor="middle", color=SSD)
    s.arrow(758, 470, 692, 470)
    # heads
    s.arrow(470, 515, 470, 553, label="h_t", lx=480, ly=540, anchor="start")
    s.box(345, 555, 250, 40, ["merge streams → LM head"], stroke=RAM, fill=RAM_BG)
    s.arrow(470, 595, 470, 621)
    s.box(370, 623, 200, 34, ["next token x_t+1"])
    s.box(650, 555, 260, 40, ["MTP head (1 layer, own MoE)"], stroke=RAM, fill=RAM_BG)
    s.path("M520,515 C580,530 620,540 648,566", color=MUTED)
    s.arrow(780, 595, 780, 621)
    s.box(680, 623, 200, 34, ["draft token x̂_t+2"], stroke=RAM, fill=RAM_BG)
    # legend
    s.box(40, 600, 18, 14, [], stroke=RAM, fill=RAM_BG, rx=3)
    s.text(64, 612, "always in RAM: 5.9 GiB", size=12)
    s.box(40, 624, 18, 14, [], stroke=SSD, fill=SSD_BG, rx=3)
    s.text(64, 636, "can live on SSD: 93 GiB", size=12)
    s.box(40, 648, 18, 14, [], stroke=STATE, fill=STATE_BG, rx=3)
    s.text(64, 660, "fixed-size recurrent state", size=12)
    s.text(40, 700, "Layer types and sizes from config.json; bytes from the oQ4e checkpoint's safetensors headers.",
           size=11.5, color=MUTED)
    s.save("arch_qwen38_flash_next.svg", "Qwen3.8-Flash-Next: 12 groups of three Gated DeltaNet layers and one Qwen Sparse "
                                         "Attention layer, each followed by a 512-expert MoE, over four gated residual "
                                         "streams, with an n-gram memory added at layer 2; experts and n-gram table can "
                                         "live on SSD.")


def qsa():
    s = SVG(960, 470, "Qwen Sparse Attention: score 4-token blocks, attend only to the best 512 of them")
    n, x0, w, gap, y = 26, 40, 24, 4, 82
    s.text(40, 66, "① indexer key for every past token (1 head, d 128)", size=12.5, color=MUTED)
    sel_blocks = {1, 4}
    for i in range(n):
        b = i // 4
        if b in sel_blocks:
            st, fl = RAM, RAM_BG
        elif i >= 24:
            st, fl = STATE, STATE_BG
        else:
            st, fl = LINE, BOX
        s.box(x0 + i * (w + gap), y, w, 26, [], stroke=st, fill=fl, rx=4)
    for b in range(6):
        bx = x0 + b * 4 * (w + gap)
        st, fl = (RAM, RAM_BG) if b in sel_blocks else (LINE, BOX)
        s.line(bx + 2, 112, bx + 4 * (w + gap) - gap - 2, 112, color=st)
        s.arrow(bx + 2 * (w + gap) - gap / 2, 114, bx + 2 * (w + gap) - gap / 2, 162, color=st)
        s.box(bx + 2, 164, 4 * (w + gap) - gap - 4, 30, [f"k̄_{b}"], stroke=st, fill=fl, size=12, rx=5)
    s.text(x0 + 24 * (w + gap), 126, "tail block", size=12, color=STATE)
    s.text(40, 216, "② average-pool each block of r = 4 tokens into one key k̄_b, then RoPE at the block's start", size=12.5,
           color=MUTED)
    # query and scores
    s.box(800, 158, 130, 42, ["query token i"], stroke=INK, fill="#ffffff")
    s.text(40, 240, "③ score every complete block:  I(i, b) = Σ over 4 heads of ReLU(⟨q_h,i , k̄_b⟩)", size=12.5,
           color=MUTED)
    s.arrow(798, 179, 712, 179, dash=True)
    s.text(755, 171, "scores", size=12, color=MUTED, anchor="middle")
    s.text(40, 274, "④ keep the top K / r = 512 blocks (K = 2,048 tokens) + the unfinished tail block", size=12.5,
           color=MUTED)
    # core attention
    s.box(40, 300, 880, 120, [], stroke=INK, fill="#ffffff", rx=10)
    s.text(60, 326, "⑤ core attention (24 Q / 2 KV heads, d 256) runs only over the selected tokens", size=13,
           weight="600")
    xs = 60
    for b in sorted(sel_blocks):
        for j in range(4):
            s.box(xs, 344, 24, 26, [], stroke=RAM, fill=RAM_BG, rx=4)
            xs += 28
        xs += 14
    for j in range(2):
        s.box(xs, 344, 24, 26, [], stroke=STATE, fill=STATE_BG, rx=4)
        xs += 28
    s.text(xs + 10, 362, "→ softmax(q·kᵀ/√d)·v → output", size=12.5)
    s.text(60, 400, "Cost per query: indexer O(n / 4) + attention O(2,048) instead of O(n).  Report, at 1M tokens: "
           "attention 7.6× faster (prefill), 4.9× (decode).", size=12, color=MUTED)
    s.text(40, 452, "The indexer is trained by distilling the dense attention distribution (max-pooled to blocks) "
           "into the block scores, then fine-tuned with sparse attention.", size=11.5, color=MUTED)
    s.save("qsa.svg", "Qwen Sparse Attention: indexer keys are average-pooled into 4-token blocks, each query scores "
                      "the blocks, keeps the top 512 plus the unfinished tail, and runs attention only over those tokens.")


def gated_residual():
    s = SVG(960, 372, "Gated residual: four residual streams, an element-wise read gate and a scalar write gate per stream")
    ys = (70, 96, 122, 148)
    for k, y in enumerate(ys):
        s.line(40, y, 920, y, color=RAM, width=2.2)
        s.text(30, y + 4, f"H{k + 1}", size=12, color=RAM, anchor="end")
    s.text(925, 113, "→", size=16, color=RAM)
    # read path
    for y in ys:
        s.line(170, y, 170, 200, color=LINE)
    s.box(110, 202, 120, 44, ["RMSNorm", "n = Norm(H)"], size=12)
    s.arrow(230, 224, 258, 224)
    s.box(260, 196, 190, 56, ["read gate (low-rank 320)", "g = σ(W_up SiLU(W_down n / 4))"], size=12)
    s.arrow(450, 224, 478, 224)
    s.box(480, 202, 150, 44, ["x = ¼ Σ_s g_s ⊙ n_s", "one 2,560-dim input"], size=12)
    s.arrow(630, 224, 658, 224)
    s.box(660, 196, 140, 56, ["block", "DeltaNet / QSA / MoE"], stroke=INK, fill="#ffffff", size=12)
    s.arrow(800, 224, 828, 224, label="y")
    s.box(830, 196, 110, 56, ["write gates", "w_s = 2σ(·)"], size=12)
    # write back
    for y in ys:
        s.path(f"M885,196 L885,{y + 4}", color=MUTED, arrow=False)
        s.text(885, y + 5, "⊕", size=15, color=INK, anchor="middle")
    s.text(875, 184, "H_s += w_s · y", size=12, color=MUTED, anchor="end")
    s.path("M170,246 C170,300 880,300 885,254", color=MUTED, dash=True)
    s.text(520, 316, "the write gates are computed from the same normalized streams n", size=12, color=MUTED,
           anchor="middle")
    s.text(40, 340, "Why: with one pre-norm stream every layer reads the same mixture. Widening to 4 streams already lowers "
           "loss (AltUp-style);", size=11.5, color=MUTED)
    s.text(40, 356, "the gates decide how each layer spends that width.", size=11.5, color=MUTED)
    s.save("gated_residual.svg", "Gated residual: the hidden state is four residual streams; each block reads a gated "
                                 "average of the normalized streams and writes its output back to each stream with a "
                                 "learned scalar gate.")


def ngram():
    s = SVG(1080, 470, "N-gram embedding: hash the last 2–3 tokens, look up 16 rows, gate them into layer 2")
    for i, t in enumerate(("x_t−2", "x_t−1", "x_t")):
        s.box(60 + i * 90, 64, 76, 34, [t], size=12.5)
    s.box(40, 140, 150, 40, ["bigram (x_t−1, x_t)"], size=12)
    s.box(205, 140, 200, 40, ["trigram (x_t−2, x_t−1, x_t)"], size=12)
    s.arrow(188, 98, 140, 138)
    s.arrow(248, 98, 290, 138)
    s.box(445, 125, 210, 70, ["hash", "⊕ of x_j × odd multiplier m_j", "(64-bit, seeded per layer)"], size=12)
    s.path("M115,180 L115,215 L470,215 L470,197", color=MUTED)
    s.arrow(405, 160, 443, 160)
    # 16 heads
    s.box(690, 60, 180, 230, [], stroke=LINE, fill="#ffffff", rx=10)
    s.text(780, 82, "16 hash heads", size=12.5, anchor="middle", weight="600")
    s.text(780, 100, "2 orders × 8 heads", size=12, anchor="middle", color=MUTED)
    for k in range(6):
        y = 112 + k * 26
        label = f"head {k + 1}: mod p_{k + 1}" if k < 5 else "…  head 16: mod p_16"
        s.box(702, y, 156, 20, [label], size=11.5, rx=4)
    s.arrow(655, 160, 688, 160)
    s.text(780, 280, "p_h: distinct primes ≈ 20 M", size=11.5, anchor="middle", color=MUTED)
    # table
    s.box(900, 60, 160, 120, ["n-gram table", "≈ 320 M rows × 160", "51 B params, 29.8 GiB", "on SSD: read 16 rows"],
          stroke=SSD, fill=SSD_BG, size=12)
    s.arrow(870, 205, 940, 182, label="16 row ids", lx=878, ly=226, anchor="start")
    s.arrow(980, 180, 980, 238, label="16 × 160 values", lx=990, ly=212, anchor="start")
    s.box(900, 240, 160, 40, ["concat → e (2,560)"], size=12)
    # injection
    s.box(40, 330, 250, 56, ["query: hidden state of", "each stream at layer 2"], stroke=RAM, fill=RAM_BG, size=12)
    s.box(340, 320, 300, 76, ["gate per stream", "σ( signed-√(⟨W_k e, q_s⟩ / √d) )", "× value W_v e"], size=12)
    s.path("M980,280 C980,305 760,305 642,345", color=MUTED)
    s.arrow(290, 358, 338, 358)
    s.box(690, 330, 370, 56, ["+ short conv (kernel 4, dilation 3)", "→ added to the 4 residual streams"], size=12)
    s.arrow(640, 368, 688, 368)
    s.text(40, 430, "Per token: no matrix work for the lookup, 16 row reads of ≈ 90 bytes at addresses known from the "
           "tokens alone,", size=12, color=MUTED)
    s.text(40, 448, "so the table can sit in host memory or on SSD and be prefetched while layer 1 runs.", size=12,
           color=MUTED)
    s.save("ngram_embedding.svg", "N-gram embedding: the bigram and trigram ending at each token are hashed into 16 "
                                  "heads with prime moduli, 16 rows are read from a 51-billion-parameter table, and the "
                                  "result is gated by the hidden state and added at layer 2.")


def expert_offload():
    s = SVG(980, 400, "One MoE layer during decode with expert offload (oMLX)")
    s.box(40, 70, 130, 50, ["router", "10 of 512"], size=12.5)
    s.arrow(170, 95, 218, 95)
    s.box(220, 60, 170, 70, ["split routes", "resident slot?"], size=12.5)
    # hit path
    s.arrow(390, 80, 448, 80, label="hits", lx=419, ly=72)
    s.box(450, 56, 220, 48, ["hits: compute on GPU now", "≈ 80 % of routes at 25 % residency"], stroke=RAM, fill=RAM_BG, size=12)
    # miss path
    s.arrow(390, 112, 448, 168, label="misses", lx=405, ly=168, anchor="end")
    s.box(450, 150, 220, 64, ["pread from the safetensors shard", "12 threads, ≤ 48 in flight", "0.4–1.9 ms per 2.8 MB expert"],
          stroke=SSD, fill=SSD_BG, size=12)
    s.arrow(670, 182, 718, 182)
    s.box(720, 158, 220, 48, ["install into a free slot", "evict lowest decayed count"], size=12)
    s.arrow(830, 158, 830, 108, label="compute the rest", lx=840, ly=136, anchor="start")
    s.arrow(670, 80, 718, 80)
    s.box(720, 56, 220, 48, ["layer output", "→ next layer's router"], stroke=INK, fill="#ffffff", size=12)
    s.box(40, 250, 900, 104, [], stroke=LINE, fill="#ffffff", rx=10)
    s.text(60, 276, "Measured on the M5 Pro (oMLX logs, per generated token, 48 layers):", size=12.5, weight="600")
    s.text(60, 298, "25 % resident: 74–92 expert misses → 200–250 MB read per token;  decode 15.5–17.8 tok/s "
           "(≈ 60 ms per token)", size=12.5)
    s.text(60, 318, "50 % resident: 17–34 misses → 47–93 MB per token;  decode 23–28 tok/s (≈ 38 ms per token)",
           size=12.5)
    s.text(60, 338, "SSD (random 2.8 MB reads, page cache bypassed): 7.3 GB/s with 1 thread, 17–18 GB/s with 12–24 "
           "threads. Reads only, no writes.", size=12.5, color=MUTED)
    s.text(40, 384, "If a read has not arrived within 0.5 ms, the GPU computes the resident routes first, so it does not "
           "sit idle (an idle Apple GPU also lowers its clock).", size=11.5, color=MUTED)
    s.save("expert_offload.svg", "During decode each MoE layer computes routes whose experts are resident immediately, "
                                 "reads missing experts from SSD in parallel, installs them in slots, and then computes "
                                 "the rest.")


def kv_reuse():
    s = SVG(980, 520, "Reusing KV across requests: prefix caching vs. block (position-independent) caching")
    segs1 = [("tools", 150), ("system", 110), ("docs A", 170), ("history", 190), ("question 1", 140)]
    segs2 = [("tools", 150), ("system", 110), ("docs B", 170), ("history", 190), ("question 2", 140)]

    def row(y, segs, colors, label):
        s.text(40, y + 22, label, size=12.5, weight="600")
        x = 190
        for (name, w), (st, fl) in zip(segs, colors):
            s.box(x, y, w - 6, 34, [name], stroke=st, fill=fl, size=12, rx=5)
            x += w
        return x

    grey = (LINE, BOX)
    row(60, segs1, [grey] * 5, "request 1")
    row(104, segs2, [grey] * 5, "request 2")
    s.text(190, 158, "Same tools, system prompt and history; only the retrieved documents differ.", size=12, color=MUTED)
    reuse, redo = (STATE, STATE_BG), (SSD, SSD_BG)
    s.text(40, 200, "Prefix caching", size=14, weight="600")
    s.text(40, 218, "(vLLM, SGLang, MLX-LM, oMLX, mlx-vlm APC, lithos-metal, Anthropic API)", size=12, color=MUTED)
    row(232, segs2, [reuse, reuse, redo, redo, redo], "request 2")
    s.text(190, 286, "Reuse stops at the first differing token: everything after docs B is recomputed, although "
           "'history' is unchanged.", size=12, color=MUTED)
    s.text(40, 330, "Block caching", size=14, weight="600")
    s.text(40, 348, "(Prompt Cache, CacheBlend / LMCache, EPIC, KVLink, …)", size=12, color=MUTED)
    row(362, segs2, [reuse, reuse, redo, reuse, redo], "request 2")
    s.text(190, 416, "'history' reuses KV computed at another position. Three problems:", size=12.5)
    s.text(206, 438, "1. position: keys were rotated for their old place → re-rotate by Δ (exact for RoPE: R(p+Δ) = R(Δ)R(p))",
           size=12, color=MUTED)
    s.text(206, 458, "2. context: those keys never attended to docs B → recompute a few tokens (CacheBlend) or train for it"
           " (KVLink, Block-Attention)", size=12, color=MUTED)
    s.text(206, 478, "3. for Gated DeltaNet layers there is no per-token KV: a recurrent state cannot be spliced at all",
           size=12, color=SSD)
    s.box(40, 494, 18, 14, [], stroke=STATE, fill=STATE_BG, rx=3)
    s.text(64, 506, "reused", size=12)
    s.box(130, 494, 18, 14, [], stroke=SSD, fill=SSD_BG, rx=3)
    s.text(154, 506, "recomputed", size=12)
    s.save("kv_reuse.svg", "Prefix caching reuses KV only up to the first differing token; block caching also reuses "
                           "later unchanged segments at a new position, which needs a position fix and a context fix, "
                           "and is not possible for recurrent DeltaNet layers.")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (arch_27b, arch_flashnext, qsa, gated_residual, ngram, expert_offload, kv_reuse):
        f()
    print("wrote", sorted(p.name for p in OUT.glob("*.svg")))
