"""S11: the paper figures, from the frozen models on the TEST groups (canvas group "Publication figures").

    python -m scripts.figures              # every figure (panels whose data is missing say so)
    python -m scripts.figures Fig2 S1      # some figures

Five main figures (one message each) and five supplementary ones. Style: Nature figure guide (Arial, 5-7 pt text,
8 pt bold panel letters, Okabe-Ito colours doubled by marker shape, keys inside the figure, no coloured text) and
Rougier et al. 2014 (one message per figure, the caption explains how to read it).
Numbers come from experiments/eval/ (S10 scores, never recomputed), s9.json / s10.json and the arrays of
scripts.figure_data (experiments/figures/data/). Writes experiments/figures/<name>.pdf (vector) and .png (600 dpi),
and the caption to experiments/figures/captions/<name>.md. Groups are never pooled: one marker shape per test group.
"""
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.text
import matplotlib.ticker
import matplotlib.transforms
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    Arc,
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Patch,
    PathPatch,
    Polygon,
    Rectangle,
)
from matplotlib.path import Path as MPath  # noqa: E402

from utils.config import ROOT  # noqa: E402

EXP = ROOT / "experiments"
OUT = EXP / "figures"
DATA = OUT / "data"
HZ = 30_000
MM = 1 / 25.4
FULL, HALF = 183 * MM, 89 * MM

GROUPS = ("p2_final", "p3_final", "p4_unseen")
GSHORT = {"p2_final": "P2", "p3_final": "P3", "p4_unseen": "P4"}
GLABEL = {"p2_final": "P2, later days", "p3_final": "P3, later days", "p4_unseen": "P4, unseen patient"}
GMARK = {"p2_final": "o", "p3_final": "s", "p4_unseen": "D"}
N = {"p2_final": (13, 161), "p3_final": (18, 562), "p4_unseen": (20, 1086)}         # recordings, Good units
VAL = {"p2_final": "p2_late", "p3_final": "p3_late"}
RATES = {"125": ("s8-ref", 240), "62.5": ("s8-s6-625hz-lr1", 480), "31.3": ("s8-s6-313hz-lr1", 960)}
# Okabe-Ito (Nature figure guide), always doubled by marker shape or line style. ŷ is one colour everywhere (trace,
# legend, Fig. 1 glyph); no red, green or magenta for x / x̂ / ŷ / units (colour-blind readers)
C = {"x": "#000000", "x_hat": "#0072B2", "y_hat": "#E69F00", "pca": "#808080", "lowrate": "#009E73", "wavelet": "#CC79A7",
     "grey": "#999999", "light": "#E6E6E6", "125": "#0072B2", "62.5": "#56B4E9", "31.3": "#A9D6F5",
     "x_hat_tint": "#EEF4FA", "x_hat_box": "#E8F1F8", "y_hat_tint": "#FDF2E0", "y_hat_line": "#F5D49B", "trace_bg": "#C8C8C8", "accent": "#D55E00"}
LS = {"x": "-", "x_hat": "-", "y_hat": "-", "pca": "--", "lowrate": ":"}
BANDS = {"delta": "δ", "theta": "θ", "alpha": "α", "beta": "β", "gamma": "γ", "high_gamma": "high γ"}
UNIT = ["#8C510A", "#E69F00", "#F3C26B"]          # unit IDs: shades of ŷ's orange (lightness differs), spikes only


def yhat_row(ax, xs, y, units, lw, y0=0.0, scale=1.0):
    """ŷ as in every figure: a thin pale-orange line and each online-sorted spike (sample, colour) in its unit's
    shade, 0.5 ms before to 1 ms after it. Fig. 1's glyph and Fig. 2a are drawn by this one function."""
    ax.plot(xs, y0 + y * scale, color=C["y_hat_line"], lw=lw * 0.5)
    for t, col in units:
        if 15 <= t < len(y) - 30:
            ax.plot(xs[t - 15:t + 30], y0 + y[t - 15:t + 30] * scale, color=col, lw=lw)
INK, MUTED = "#000000", "#555555"
CAPTIONS = {}

plt.rcParams.update({
    "font.family": "Arial", "mathtext.fontset": "custom", "mathtext.it": "Arial:italic", "mathtext.rm": "Arial",
    "mathtext.bf": "Arial:bold", "font.size": 6.5, "axes.titlesize": 6.5, "axes.labelsize": 6.5,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "lines.linewidth": 1.0, "axes.titleweight": "normal", "axes.titlelocation": "left", "axes.titlepad": 4,
    "legend.frameon": False, "legend.handlelength": 1.4, "legend.borderaxespad": 0.2, "legend.labelspacing": 0.3,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "axes.edgecolor": INK, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.labelpad": 2,
})


# ------------------------------------------------------------------ data access

def js(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ev(run, g):
    return js(EXP / "eval" / f"s10_{run}_200000_{g}.json")


def base(src, frame, g):
    return js(EXP / "eval" / f"{src}_F{frame}_K32_{g}.json")


def get(d, path):
    for p in path:
        d = d[p]
    return d


S10, S9 = js(EXP / "s10.json"), js(EXP / "s9.json")


def arm(stage, g, a, m):
    """(mean, sd) of a metric over the 5 seeds of an arm."""
    r = stage["groups"][g]["last_step"][a][m]
    return r["mean"], r["sd"] or 0.0


def seeds(g, path, arm_="s8-ref"):
    return np.array([get(ev(f"{arm_}-s{s}", g), path) for s in range(1, 6)], float)


def npz(name):
    p = DATA / f"{name}.npz"
    return (np.load(p, allow_pickle=True), js(DATA / f"{name}.json")) if p.exists() else (None, None)


# ------------------------------------------------------------------ style helpers

def letter(ax, l, title=None, dx=-0.16, dy=1.0):
    """8 pt bold panel letter outside the top-left corner; a short finding as the panel title."""
    ax.text(dx, dy, l, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")
    if title:
        ax.set_title(title, loc="left", fontsize=6.5, pad=4)


def ygrid(ax):
    ax.grid(axis="y", color="#E6E6E6", lw=0.4)
    ax.set_axisbelow(True)


def pending(ax, what):
    ax.text(0.5, 0.5, f"awaiting data\n({what})", transform=ax.transAxes, ha="center", va="center", fontsize=6,
            color=MUTED)
    ax.set_xticks([]), ax.set_yticks([])
    ax.pending = True


def group_handles(col="#000000"):
    return [Line2D([], [], marker=GMARK[g], ls="", color=col, ms=3.2, label=GLABEL[g]) for g in GROUPS]


def on_or_above(per):
    """Per test group (never pooled): share of units sorting as well or better on x̂ than on x."""
    out = {}
    for g in GROUPS:
        x_ = np.array([v if v is not None else np.nan for v in per[g]["x"]], float)
        y_ = np.array([v if v is not None else np.nan for v in per[g]["x_hat"]], float)
        ok = ~np.isnan(x_) & ~np.isnan(y_)
        out[g] = float(np.mean(y_[ok] >= x_[ok]))
    return out


def save(fig, name, caption):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "captions").mkdir(exist_ok=True)
    (OUT / "captions" / f"{name}.md").write_text(caption.strip() + "\n", encoding="utf-8")
    CAPTIONS[name] = caption.strip()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ Fig. 1 · the codec

def smooth(pts, n=24):
    """Catmull-Rom curve through the points (for the head outline)."""
    p = np.asarray(pts, float)
    p = np.vstack([p[0], p, p[-1]])
    out = []
    for i in range(1, len(p) - 2):
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 +
                          (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.vstack(out + [p[-2]])


def human(ax, x0, ytop, sc, ybottom):
    """The participant of Fig. 1a 'Current' in FIGURES.html, all vector, drawn in that SVG's own coordinates and mapped
    once (mirrored so V1 and the connector face the encoder; a left-facing head shows the right hemisphere, where the
    array is): the head silhouette (assets/head_current.npz), the brain (assets/brain_c.npz, its line drawing traced
    with potrace), the Utah array at V1 with the same tilt and 96-electrode grid, its wire to the connector on the skull,
    the dashed V1 circle. sc: data units per SVG unit; the bust is cut at ybottom. Returns the connector's cable end."""
    M = matplotlib.transforms.Affine2D().scale(-sc, -sc).translate(x0 + 209.8 * sc, ytop + 26.7 * sc)   # SVG -> data
    T = lambda t: t + M + ax.transData
    clip = Rectangle((x0 - 5, ybottom), 300 * sc, 300 * sc, transform=ax.transData, visible=False)
    ax.add_patch(clip)
    h = np.load(OUT / "assets" / "head_current.npz")
    ax.add_patch(PathPatch(MPath(h["verts"], h["codes"]), fc="#fafafa", ec="#b3b3b3", lw=0.6, joinstyle="round",
                           transform=T(matplotlib.transforms.Affine2D()), zorder=2)).set_clip_path(clip)
    d = np.load(OUT / "assets" / "brain_c.npz")
    W, H = d["size"]
    B = T(matplotlib.transforms.Affine2D().scale(129.48 / W, 92.04 / H).translate(58.72, 31.36))
    for name, fc in (("silhouette", "white"), ("cerebellum", "#E3E3E3"), ("outline", "#111111")):
        ax.add_patch(PathPatch(MPath(d[f"{name}/verts"], d[f"{name}/codes"]), fc=fc, ec="none", transform=B, zorder=3))
    I = T(matplotlib.transforms.Affine2D())
    ax.add_patch(PathPatch(MPath([(68, 68.7), (78.8, 39.0), (87.4, 106.76), (77.4, 46.76)],
                                 [MPath.MOVETO] + [MPath.CURVE4] * 3), fc="none", ec="#111111", lw=0.4,
                           transform=I, zorder=4))
    A = T(matplotlib.transforms.Affine2D().scale(0.52, 1).skew_deg(-10, 0).rotate_deg(-8).translate(68, 75))
    for x, y, fc, alpha, ec in ((-8.8, -9.2, "#000000", 0.22, "none"), (-9.9, -10.3, "#5b6270", 1, "none"),
                                (-11.0, -11.0, "#dfe3e9", 1, "#111111")):
        ax.add_patch(FancyBboxPatch((x, y), 22, 22, boxstyle="round,pad=0,rounding_size=1.1", fc=fc, alpha=alpha,
                                    ec=ec, lw=0.3, transform=A, zorder=6))
    g = np.linspace(-9.34, 9.34, 10)
    for i, u in enumerate(g):
        for j, v in enumerate(g):
            if (i, j) not in ((0, 0), (0, 9), (9, 0), (9, 9)):
                ax.add_patch(Circle((u, v), 0.58, fc="#2a3140", ec="none", transform=A, zorder=7))
    P = T(matplotlib.transforms.Affine2D().rotate_deg_around(76.4, 41.76, -38))
    for x, y, w, hh, r in ((66.7, 31.6, 19.3, 12.0, 3), (70.0, 28.0, 12.9, 4.1, 2)):
        ax.add_patch(FancyBboxPatch((x, y), w, hh, boxstyle=f"round,pad=0,rounding_size={r}", fc="#111111", ec="none",
                                    transform=P, zorder=6))
    c = M.transform((67.8, 74.6))
    ax.add_patch(Circle(c, 13 * sc, fc="none", ec="#777777", lw=0.5, ls=(0, (2.5, 2)), zorder=5))
    return M.transform((73.4, 34.8)), c, 13 * sc


def schematic(ax, snip=None, units=None):
    """Panel a. With data, the signal glyphs are 100 ms of the recording of b (x and x̂ on one scale).
    units: (sample in snip, colour) of ŷ's online-sorted spikes: ŷ is drawn grey and each spike in its unit colour."""
    rng = np.random.default_rng(3)
    ax.set_xlim(0, 183), ax.set_ylim(0, 58), ax.set_aspect("equal"), ax.axis("off")

    def card(x, y, w, h, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=3.5", fc=fc, ec="none"))

    def trapezoid(x, c, w, h_in, h_out, col, text):
        pts = [(x, c - h_in / 2), (x + w, c - h_out / 2), (x + w, c + h_out / 2), (x, c + h_in / 2)]
        ax.add_patch(Polygon(pts, closed=True, fc=col, ec=col, lw=2.5, joinstyle="round"))
        ax.text(x + w / 2, c, text, ha="center", va="center", color="white", fontsize=7, fontweight="bold",
                linespacing=1.0)

    def glyph(x0, y0, w, h, sig, col, lw=0.6, scale=None):
        ax.plot(np.linspace(x0, x0 + w, len(sig)), y0 + sig / (scale or np.abs(sig).max()) * h / 2, color=col,
                lw=lw)

    def arrow(a, b, col=INK, lw=0.8, **kw):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=6, color=col, lw=lw, shrinkA=0,
                                     shrinkB=0, **kw))

    if snip is None:                                            # no data yet: seeded illustrative glyphs
        n = 700
        slow = np.convolve(np.cumsum(rng.normal(0, 1, n)), np.ones(25) / 25, "same")
        slow = (slow - slow.mean()) / np.abs(slow - slow.mean()).max()
        spikes = np.zeros(n)
        for t in (90, 230, 260, 420, 560, 640):
            spikes[t:t + 12] -= np.r_[np.hanning(8), -np.hanning(4) * 0.35] * 9
        noise = rng.normal(0, 0.45, n)
        x = slow * 4 + spikes + noise
        snip = (x, slow * 4 + spikes * 0.97 + np.convolve(noise, np.ones(3) / 3, "same"), spikes)
    x, x_hat, y_hat = snip
    mu = x.mean()
    x, x_hat = x - mu, x_hat - mu
    sc = np.abs(np.r_[x, x_hat]).max()
    card(1, 2, 97, 54, C["x_hat_tint"])
    card(101, 2, 81, 54, C["y_hat_tint"])
    ax.text(5, 50.5, "Implant", fontsize=7.5, fontweight="bold")
    ax.text(105, 50.5, "Receiver", fontsize=7.5, fontweight="bold")
    (px, py), v1, rc = human(ax, 3, 47.5, 0.2, 13.5)
    ex = 51.5                                                   # the connector's cable into the encoder's input
    ax.add_patch(PathPatch(MPath([(px, py), (px + 5, py + 6), (ex - 9, 33.5), (ex - 2.2, 33.5)],
                                 [MPath.MOVETO] + [MPath.CURVE4] * 3), fc="none", ec="#3a3a3a", lw=0.8,
                           capstyle="round", zorder=3))
    arrow((ex - 2.3, 33.5), (ex, 33.5))
    ax.text(v1[0] + rc + 0.8, v1[1], "V1", ha="left", va="center", fontsize=6, fontweight="bold", zorder=8)
    glyph(4, 9.5, 40, 6, x, INK, 0.4, sc)
    ax.text(24, 4.5, "96 electrodes · 30 kHz", ha="center", va="center", fontsize=6)
    trapezoid(51.5, 33.5, 16, 22, 9, C["x_hat"], "Encoder")
    ax.text(59.5, 20.5, "causal,\nstreaming", ha="center", va="top", fontsize=6, linespacing=1.05)
    arrow((68.5, 33.5), (72, 33.5))
    for k in range(5):
        for m in range(8):
            ax.add_patch(FancyBboxPatch((72.5 + k * 3.0, 27.5 + m * 1.5), 2.5, 1.2, ec="none",
                                        boxstyle="round,pad=0,rounding_size=0.3",
                                        fc=matplotlib.cm.Blues(rng.uniform(0.3, 1))))
    ax.text(79.75, 24.5, "tokens", ha="center", va="top", fontsize=6.5, fontweight="bold")
    ax.text(79.75, 21, "125 Hz", ha="center", va="top", fontsize=6)
    for r_ in (2.2, 4.0, 5.8):
        ax.add_patch(Arc((91, 33.5), r_, r_ * 1.4, theta1=-50, theta2=50, color=INK, lw=0.7))
    arrow((94, 33.5), (104, 33.5), INK, 0.9)
    arrow((104.5, 33.5), (108.5, 42.5)), arrow((104.5, 33.5), (108.5, 18.5))
    trapezoid(109, 42.5, 20, 8, 16, C["x_hat"], "Decoder")
    trapezoid(109, 18.5, 20, 8, 16, C["y_hat"], "Spike\ndecoder")
    arrow((130, 42.5), (135, 42.5)), arrow((130, 18.5), (135, 18.5))
    glyph(136, 42.5, 42, 10, x_hat, C["x_hat"], 0.4, sc)
    if units is None:
        glyph(136, 18.5, 42, 10, y_hat, C["y_hat"], 0.5)
    else:
        yhat_row(ax, np.linspace(136, 178, len(y_hat)), y_hat, units, 0.7, 18.5, 5 / np.abs(y_hat).max())
    ax.text(157, 49, "$\\hat{x}$: the whole recording", ha="center", fontsize=6.5)
    ax.text(157, 25, "$\\hat{y}$: spikes only", ha="center", fontsize=6.5)
    ax.text(157, 9.5, "LFP, MUA, sorting · online unit IDs",
            ha="center", va="top", fontsize=6, linespacing=1.05)


def glyph_snip(ex):
    """100 ms of row 0 around the spike of Fig. 1c: x, x̂, ŷ (None while the examples are missing)."""
    if ex is None:
        return None
    c0 = spike_c(ex)
    g = slice(max(c0 - 1500, 0), min(c0 + 1500, len(ex["0/x"])))
    return ex["0/x"][g], ex["0/x_hat"][g], ex["0/y_hat"][g]


def spike_c(ex):
    """The isolated NIX Good spike (no other within the 5 ms window) nearest the middle of row 0 whose trough on the
    original is 8–12σ: chosen on x only."""
    sig, bx, nix, n = float(ex["sigma_bp"][0]), ex["0/bp_x"], ex["0/nix_t"], len(ex["0/x"])
    ok = [int(t) for t in nix if 600 <= t < n - 900 and -12 <= (bx[t - 15:t + 15] / sig).min() <= -8
          and ((nix > t - 60) & (nix < t + 120)).sum() == 1]
    return min(ok or [int(t) for t in nix if 600 <= t < n - 900], key=lambda t: abs(t - n // 2))


def fig1():
    ex, _ = npz("examples")
    fig = plt.figure(figsize=(FULL, 112 * MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1], hspace=0.12)
    ax = fig.add_subplot(gs[0])
    schematic(ax, glyph_snip(ex))
    ax.text(0.0, 1.0, "a", transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
    sub = gs[1].subgridspec(3, 3, width_ratios=[2.2, 0.8, 1.1], height_ratios=[1, 1, 1], wspace=0.28, hspace=0.12)
    if ex is None:
        for k, l in ((0, "b"), (1, "c"), (2, "d")):
            a = fig.add_subplot(sub[:, k])
            pending(a, "examples")
            letter(a, l)
    else:
        sig = float(ex["sigma_bp"][0])
        x, xh, yh, bx = ex["0/x"], ex["0/x_hat"], ex["0/y_hat"], ex["0/bp_x"]
        nix = ex["0/nix_t"]
        t = np.arange(len(x)) / HZ
        rows = [(x, C["x"], "$x$"), (xh, C["x_hat"], "$\\hat{x}$"), (yh / sig, C["y_hat"], "$\\hat{y}$")]
        axes = []
        for i, (v, col, lab) in enumerate(rows):
            a = fig.add_subplot(sub[i, 0])
            a.plot(t, v, color=col, lw=0.35)
            a.set_xlim(0, 1)
            a.set_yticks([])
            a.spines["left"].set_visible(False)
            a.text(-0.01, 0.5, lab, transform=a.transAxes, ha="right", va="center", fontsize=7)
            if i < 2:
                a.set_xticks([])
                a.spines["bottom"].set_visible(False)
            axes.append(a)
        axes[-1].set_xlabel("time (s)")
        letter(axes[0], "b", "1 s of a P4 recording (patient never seen in training)", dx=-0.06)
        c = spike_c(ex)
        w = slice(c - 45, c + 105)
        a = fig.add_subplot(sub[:, 1])
        tt = (np.arange(w.start, w.stop) - c) / 30
        a.plot(tt, bx[w] / sig, color=C["x"], lw=1.0, label="bp($x$)")
        a.plot(tt, ex["0/bp_x_hat"][w] / sig, color=C["x_hat"], lw=1.0, ls="--", label="bp($\\hat{x}$)")
        a.plot(tt, yh[w] / sig, color=C["y_hat"], lw=1.0, ls=":", label="$\\hat{y}$")
        a.axhline(-5, color=C["grey"], lw=0.5, ls="--")
        a.text(tt[0], -4.8, "−5σ", fontsize=5.5, ha="left", va="bottom", color=MUTED)
        a.set_xlabel("ms")
        a.set_ylabel("σ")
        a.legend(loc="lower right", fontsize=5.5, handlelength=1.2)
        letter(a, "c", "one spike, 5 ms", dx=-0.32)
        a = fig.add_subplot(sub[:, 2])
        from utils.dsp import spike_mask
        m = spike_mask(np.zeros(len(nix), np.int64), nix, 1, 0, len(x))[0]
        ww = slice(c - 600, c + 900)
        t2 = (np.arange(ww.start, ww.stop) - c) / 30
        top = -7.0                                                  # stack the traces by their own range
        for v, col, lab in ((bx / sig, C["x"], "bp($x$)"), (m * 6, C["lowrate"], "mask $m$"),
                            (m * bx / sig, C["y_hat"], "target $y=m\\cdot$bp($x$)")):
            seg = v[ww]
            off = top - seg.max()
            a.plot(t2, seg + off, color=col, lw=0.6)
            a.text(t2[0], top + 1, lab, fontsize=5.8, va="bottom")
            top = off + seg.min() - 9
        a.set_ylim(top + 7, 0)
        a.set_yticks([]), a.spines["left"].set_visible(False)
        a.set_xlabel("ms")
        letter(a, "d", "what $\\hat{y}$ is trained to output", dx=-0.13)
    save(fig, "Fig1_codec", """
**Figure 1 | A causal codec turns 30 kHz recordings into 125 Hz tokens and returns the whole recording and a spikes-only trace.**
**a**, The encoder runs on the implant: each electrode's signal is compressed, causally and frame by frame, into
32 numbers every 8 ms. At the receiver, one decoder rebuilds the whole recording ($\\hat x$) and a second decoder outputs
only the spikes of the NIX-curated (Good) units ($\\hat y$). Illustration (not to scale): a participant with the Utah array (10 × 10, 400 µm pitch, 96 electrodes) in V1, near the occipital pole, cabled through its connector on the skull. Signal glyphs: the 100 ms of the recording in b around the spike in c ($x$ and $\\hat x$ on one scale). **b**, One second of a recording
of patient P4, who was never seen in training: original $x$, reconstruction $\\hat x$ and spike output $\\hat y$ (each scaled to
its own range). **c**, One spike, 5 ms, in the spike band (500–7500 Hz), in units of σ (the robust noise SD of bp($x$)); the
dashed line is the −5σ detection threshold. **d**, How the training target of $\\hat y$ is built: the NIX Good spike times
give a mask $m$ (2 ms Tukey windows), and the target is the recording's own spike band inside the mask. NIX waveforms are
never a target (they are whitened). Data b–d: 125 Hz model, seed 1, test group p4_unseen.""")


# ------------------------------------------------------------------ Fig. 2 · spikes survive (headline)

def fig2():
    mg, mmeta = npz("margin")
    wf, _ = npz("waveforms")
    fig = plt.figure(figsize=(FULL, 118 * MM))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.25, 1.25], height_ratios=[1, 1], hspace=0.55, wspace=0.42)
    # a · retention per group
    a = fig.add_subplot(gs[0, 0])
    srcs = [("x_hat", "codec $\\hat{x}$"), ("y_hat", "spike decoder $\\hat{y}$"), ("pca", "PCA"),
            ("lowrate", "plain sampling")]
    for i, g in enumerate(GROUPS):
        for j, (k, _) in enumerate(srcs):
            xpos = i + (j - 1.5) * 0.18
            if k in ("x_hat", "y_hat"):
                mu, sd = arm(S10, g, "s8-ref", f"retention_{k}")
                a.errorbar(xpos, mu, sd, marker=GMARK[g], color=C[k], ms=3.5, capsize=1.2, lw=0.7)
            else:
                v = base(k, 240, g)["retention_x_hat"]["retention"]
                a.plot(xpos, v, marker=GMARK[g], color=C[k], ms=3.5, ls="")
    a.axhline(1, color=INK, lw=0.5)
    a.axhline(0.95, color=C["grey"], lw=0.5, ls="--")
    a.text(2.45, 0.955, "0.95", fontsize=5.5, color=MUTED, ha="right", va="bottom")
    a.set_xticks(range(3), [GSHORT[g] for g in GROUPS])
    a.set_ylim(0.4, 1.02)
    a.set_ylabel("retention (spikes kept)")
    a.legend(handles=[Line2D([], [], color=C[k], marker="o", ls="", ms=3.5, label=lab) for k, lab in srcs],
             loc="center left", bbox_to_anchor=(0, 0.66), fontsize=5.5)
    ygrid(a)
    rx = [arm(S10, g, "s8-ref", "retention_x_hat")[0] * 100 for g in GROUPS]
    letter(a, "a", f"the codec keeps {min(rx):.0f}–{max(rx):.0f} %")
    # b · retention vs unit SNR
    a = fig.add_subplot(gs[0, 1])
    edges = np.array([6, 7.5, 9, 10.5, 12, 15, 18, 25, 40, 80])
    for k, ls in (("x_hat", "-"), ("y_hat", "--")):
        for g in GROUPS:
            u = ev("s8-ref-s1", g)["units"]
            snr = np.array([v["amp_sigma"] for v in u])
            ret = np.array([v.get(f"{k}_retention") if v.get(f"{k}_retention") is not None else np.nan for v in u])
            if k == "x_hat" and g == "p4_unseen":
                a.scatter(snr, ret, s=1.5, color=C[k], alpha=0.15, lw=0, zorder=1)
            b = np.digitize(snr, edges)
            mids, med = [], []
            for i_ in range(1, len(edges)):
                sel = (b == i_) & np.isfinite(ret)
                if sel.sum() >= 5:
                    mids.append(np.sqrt(edges[i_ - 1] * edges[i_])), med.append(np.median(ret[sel]))
            a.plot(mids, med, color=C[k], ls=ls, marker=GMARK[g], ms=2.8, lw=0.9, zorder=3)
    a.set_xscale("log")
    a.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    a.set_xticks([6, 10, 20, 40, 80])
    a.set_ylim(0.5, 1.02)
    a.axhline(0.95, color=C["grey"], lw=0.5, ls="--")
    a.set_xlabel("unit SNR (peak-to-peak / σ)")
    a.set_ylabel("retention per unit (median)")
    a.legend(handles=[Line2D([], [], color=C["x_hat"], label="$\\hat{x}$"),
                      Line2D([], [], color=C["y_hat"], ls="--", label="$\\hat{y}$")] + group_handles(),
             loc="lower right", fontsize=5.5, ncol=1)
    ygrid(a)
    letter(a, "b", "clear units keep every spike")
    # c · retention by trough depth
    a = fig.add_subplot(gs[0, 2])
    if mg is None:
        pending(a, "margin")
    else:
        lab = ["5–5.25", "5.25–5.5", "5.5–6", "6–7", "7–9", "9–12", "≥12"]
        xs = np.arange(len(lab))
        for g in GROUPS:
            found = mg[f"{g}/codec_s1/found"]
            share = found / found.sum()
            if g == "p4_unseen":
                a2 = a.twinx()
                a2.bar(xs, share, 0.7, color=C["light"], zorder=0)
                a2.set_ylim(0, 1.0)
                a2.set_yticks([0, 0.1, 0.2, 0.3])
                a2.set_ylabel("share of spikes (P4)", fontsize=5.5, color=MUTED)
                a2.tick_params(labelsize=5, colors=MUTED)
                a2.spines["right"].set_visible(True), a2.spines["right"].set_color(MUTED)
                a.set_zorder(a2.get_zorder() + 1), a.patch.set_visible(False)
            for k, ls in (("x_hat", "-"), ("y_hat", "--")):
                r = np.array([mg[f"{g}/codec_s{s}/{k}"] / np.maximum(mg[f"{g}/codec_s{s}/found"], 1)
                              for s in range(1, 6)])
                a.plot(xs, r.mean(0), color=C[k], ls=ls, marker=GMARK[g], ms=2.8, lw=0.9)
            p = mg[f"{g}/pca_F240/x_hat"] / np.maximum(mg[f"{g}/pca_F240/found"], 1)
            a.plot(xs, p, color=C["pca"], ls=LS["pca"], marker=GMARK[g], ms=2.5, lw=0.7)
        a.axvspan(-0.5, 1.5, color="#FFF4E0", zorder=0)
        a.axhline(0.95, color=C["grey"], lw=0.5, ls="--")
        a.set_xticks(xs, lab, rotation=40, ha="right")
        a.set_xlabel("trough depth on the original (σ); threshold = 5")
        a.set_ylabel("retention")
        a.set_ylim(0, 1.03)
        a.text(0.5, 0.25, "just over\nthe threshold", fontsize=5.5, ha="center", color=MUTED)
        ygrid(a)
    letter(a, "c", "what is lost only just crossed −5σ")
    # d · waveforms
    sub = gs[1, :2].subgridspec(2, 4, wspace=0.15, hspace=0.35)
    if wf is None:
        a = fig.add_subplot(sub[:, :])
        pending(a, "waveforms")
        letter(a, "d")
    else:
        snr = wf["snr"]
        o = np.argsort(snr)
        pick = o[(np.linspace(0.05, 0.98, 8) * (len(o) - 1)).astype(int)][::-1]
        t = (np.arange(61) - 21) / 30
        for n_, u in enumerate(pick):
            a = fig.add_subplot(sub[n_ // 4, n_ % 4])
            m, s = wf["bp_x_mean"][u], wf["bp_x_sd"][u]
            a.fill_between(t, m - s, m + s, color=C["x"], alpha=0.10, lw=0)
            a.plot(t, m, color=C["x"], lw=0.9)
            a.plot(t, wf["bp_x_hat_mean"][u], color=C["x_hat"], lw=0.9, ls="--")
            a.plot(t, wf["y_hat_mean"][u], color=C["y_hat"], lw=0.9, ls=":")
            a.set_title(f"{snr[u]:.0f}σ · r {wf['wave_r'][u]:.2f}", fontsize=5.5, pad=1.5, loc="center")
            a.set_xticks([0, 1]) if n_ // 4 == 1 else a.set_xticks([])
            a.tick_params(labelsize=5)
            if n_ % 4:
                a.set_yticks([])
            if n_ == 0:
                letter(a, "d", None, dx=-0.45, dy=1.28)
                a.text(-0.2, 1.28, "spike shapes are kept, from large to small units (P4)", transform=a.transAxes,
                       fontsize=6.5, va="bottom")
                a.legend(handles=[Line2D([], [], color=C["x"], label="$x$"),
                                  Line2D([], [], color=C["x_hat"], ls="--", label="$\\hat{x}$"),
                                  Line2D([], [], color=C["y_hat"], ls=":", label="$\\hat{y}$")],
                          loc="lower right", fontsize=5, handlelength=1.3, borderpad=0.1, labelspacing=0.15)
            if n_ == 4:
                a.set_xlabel("ms", fontsize=5.5)
                a.set_ylabel("σ", fontsize=5.5)
    # e · detection
    a = fig.add_subplot(gs[1, 2])
    srcs = [("x", "original $x$"), ("x_hat", "$\\hat{x}$"), ("y_hat", "$\\hat{y}$"), ("pca", "PCA")]
    for i, g in enumerate(GROUPS):
        for j, (k, _) in enumerate(srcs):
            xpos = i + (j - 1.5) * 0.19
            if k in ("x_hat", "y_hat"):
                f1 = seeds(g, (f"detect_{k}", "f1"))
                d = ev("s8-ref-s1", g)[f"detect_{k}"]
                a.errorbar(xpos, f1.mean(), f1.std(ddof=1), marker=GMARK[g], color=C[k], ms=3.5, capsize=1, lw=0.7)
            else:
                d = base("pca", 240, g)["detect_x_hat"] if k == "pca" else ev("s8-ref-s1", g)["detect_x"]
                a.plot(xpos, d["f1"], marker=GMARK[g], color=C[k], ms=3.5, ls="")
            a.plot([xpos] * 2, [d["recall"], d["precision"]], color=C[k], lw=0.5, alpha=0.6)
    a.set_xticks(range(3), [GSHORT[g] for g in GROUPS])
    a.set_ylim(0.3, 1.0)
    a.set_ylabel("detection F1")
    a.legend(handles=[Line2D([], [], color=C[k], marker="o", ls="", ms=3.5, label=lab) for k, lab in srcs],
             loc="lower left", fontsize=5.5, ncol=2)
    ygrid(a)
    letter(a, "e", "detection beats the original")
    save(fig, "Fig2_spikes", """
**Figure 2 | At 125 Hz the codec keeps 92–96 % of the spikes the original recording reveals; what it loses are spikes that only just cross the detection threshold.**
**a**, Retention (of the NIX Good spikes detected on the original, the share also detected on the output) per test group,
for the codec reconstruction $\\hat x$, the spike decoder $\\hat y$, PCA with the same number of values per 8 ms (32) and
plain sampling at the same rate. Codec: mean ± SD over 5 training seeds. **b**, Median retention per unit against the
unit's SNR (6–80σ bins with ≥ 5 units); dots: every P4 unit ($\\hat x$). Units above ~12σ keep all their spikes.
**c**, Retention by how deep each spike's trough goes on the original (σ units; the detector threshold is 5σ); grey bars:
share of P4 spikes in each bin. The loss is concentrated in the first 0.5σ above the threshold, where background noise
decided whether the spike crossed at all; the codec removes that noise. Codec: mean of 5 seeds; PCA dashed.
**d**, Mean spike waveforms of eight P4 units from the clearest to the smallest (titles: SNR and the median per-spike
correlation of bp($\\hat x$) with bp($x$)); solid: bp($x$) ± SD, dashed: bp($\\hat x$), dotted: $\\hat y$. **e**, Detection F1 (marker) with recall and
precision (ends of the line) on the original, the two codec outputs and PCA. Precision counts a detection as correct when
it matches any NIX unit (Good or Bad); recall is against Good units. Detector everywhere: −5σ crossings in 500–7500 Hz,
σ from the original, match ±0.5 ms. Test groups (recordings / Good units): P2 13 / 161, P3 18 / 562, P4 20 / 1 086
(P4: a patient never used in training).""")


# ------------------------------------------------------------------ Fig. 3 · how far the rate can drop

def codec(m):
    """[group, rate, (mean, sd over 5 seeds)]; a tuple m is a path into the eval JSONs."""
    if isinstance(m, tuple):
        return np.array([[(lambda v: (v.mean(), v.std(ddof=1)))(seeds(g, m, RATES[r][0])) for r in RATES]
                         for g in GROUPS])
    return np.array([[arm(S10, g, RATES[r][0], m) for r in RATES] for g in GROUPS])


def baseline(src, path):
    return np.array([[get(base(src, RATES[r][1], g), path) for r in RATES] for g in GROUPS])


def fig3():
    rates = list(RATES)
    xs = np.arange(len(rates))

    def draw(a, v, k, sd=None, ms=3.2, lw=1.0):
        for i, g in enumerate(GROUPS):
            off = (i - 1) * 0.07
            if sd is not None:
                a.errorbar(xs + off, v[i], sd[i], color=C[k], ls=LS[k], marker=GMARK[g], ms=ms, lw=lw, capsize=1)
            else:
                a.plot(xs + off, v[i], color=C[k], ls=LS[k], marker=GMARK[g], ms=ms * 0.85, lw=lw * 0.8)

    def axis(a, last=True):
        a.set_xticks(xs, rates if last else [""] * 3)
        a.set_xlim(-0.35, 2.35)
        ygrid(a)

    fig = plt.figure(figsize=(FULL, 82 * MM))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.6, 1, 1, 1], hspace=0.5, wspace=0.5)
    a = fig.add_subplot(gs[:, 0])
    for k in ("x_hat", "y_hat"):
        v = codec(f"retention_{k}")
        draw(a, v[..., 0], k, v[..., 1])
    for k in ("pca", "lowrate"):
        draw(a, baseline(k, ("retention_x_hat", "retention")), k)
    a.axhline(1, color=INK, lw=0.5)
    a.set_ylim(-0.02, 1.04)
    a.set_xlabel("token rate (Hz)")
    a.set_ylabel("retention (spikes kept)")
    axis(a)
    a.legend(handles=[Line2D([], [], color=C[k], ls=LS[k], label=lab) for k, lab in
                      (("x_hat", "codec $\\hat{x}$"), ("y_hat", "spike decoder $\\hat{y}$"), ("pca", "PCA, same budget"),
                       ("lowrate", "plain sampling"))] + group_handles(),
             loc="center right", bbox_to_anchor=(1.02, 0.47), fontsize=5.5)
    letter(a, "a", "codec degrades gently, linear collapses", dx=-0.2)
    small = [("f1_y_hat", "y_hat", "detection F1", ("detect_x_hat", "f1"), ("detect_x", "f1")),
             ("sorting_acc", "x_hat", "sorting accuracy", None, ("sorting_x", "acc_median")),
             ("wave_r", "x_hat", "waveform r", ("waveform", "all", "r_median"), None),
             ("mua_r", "x_hat", "MUA r", ("mua_r_median",), None),
             ("lfp_high_gamma_r", "x_hat", "LFP r (high γ)", ("lfp_r_median", "high_gamma"), None),
             (("retention_x_hat", "units", "pct_retention_ge_0.95"), "x_hat", "% units keeping ≥ 95 %", None,
              None)]
    for i, (m, k, title, bpath, ceil) in enumerate(small):
        a = fig.add_subplot(gs[i // 3, 1 + i % 3])
        if ceil:
            c = [get(ev("s8-ref-s1", g), ceil) for g in GROUPS]
            a.axhspan(min(c), max(c), color=C["light"], lw=0)
        if bpath:
            draw(a, baseline("pca", bpath), "pca", ms=2.6)
        v = codec(m)
        draw(a, v[..., 0], k, v[..., 1], ms=2.8)
        axis(a, i // 3 == 1)
        if i // 3 == 1:
            a.set_xlabel("token rate (Hz)")
        a.set_title(title, loc="left", fontsize=6.5, pad=3)
        a.text(-0.32, 1.0, "bcdefg"[i], transform=a.transAxes, fontsize=8, fontweight="bold", va="bottom")
    save(fig, "Fig3_rate", """
**Figure 3 | Halving the token rate to 62.5 Hz costs about 3 points of retention and keeps unit sorting; at 31.3 Hz every feature degrades; linear compression at the same budgets loses most spikes.**
**a**, Retention against token rate for the codec ($\\hat x$, $\\hat y$; mean ± SD over 5 seeds per rate), PCA with the same
32 values per frame, and plain sampling at the same number of values per second. One marker shape per test group; groups
are never averaged. **b–g**, Detection F1 of $\\hat y$, offline sorting accuracy of $\\hat x$ (SpikeForest accuracy per NIX
Good unit, median), waveform correlation, MUA correlation and LFP band-envelope correlation (high γ, 80–200 Hz; δ,
1–4 Hz is r ≥ 0.999 everywhere and not shown) and the share of units that keep ≥ 95 % of their spikes. Grey band: the original recording, across the three groups (b, c). PCA dashed where it applies; sorting was not run
on the baselines. At 15.6 Hz the codec collapses (S6, validation, 15 k steps; not part of the test).""")


# ------------------------------------------------------------------ Fig. 4 · LFP, MUA, sorting, new patient

def fig4():
    cr, _ = npz("channel_r")
    so, smeta = npz("sorting")
    ex, _ = npz("examples")
    fig = plt.figure(figsize=(FULL, 112 * MM))
    gs = fig.add_gridspec(2, 3, hspace=0.6, wspace=0.42, width_ratios=[1.35, 1, 1])
    # a · LFP bands
    a = fig.add_subplot(gs[0, 0])
    if cr is None:
        pending(a, "channel_r")
    else:
        for i, b in enumerate(BANDS):
            for j, g in enumerate(GROUPS):
                xpos = i + (j - 1) * 0.26
                v = cr[f"{g}/codec_125/{b}"].ravel()
                v = v[np.isfinite(v)]
                q = np.percentile(v, [10, 25, 50, 75, 90])
                a.plot([xpos] * 2, [q[0], q[4]], color=C["x_hat"], lw=0.5)
                a.add_patch(Rectangle((xpos - 0.09, q[1]), 0.18, q[3] - q[1], fc="white", ec=C["x_hat"], lw=0.6))
                a.plot(xpos, q[2], marker=GMARK[g], color=C["x_hat"], ms=2.6)
                p = np.nanmedian(cr[f"{g}/pca_F240/{b}"])
                a.plot([xpos - 0.11, xpos + 0.11], [p, p], color=C["pca"], lw=1.0)
        a.set_xticks(range(6), list(BANDS.values()))
        lo = min(np.nanpercentile(cr[f"{g}/codec_125/{b}"], 10) for g in GROUPS for b in BANDS)
        a.set_ylim(max(0.85, lo - 0.01), 1.003)
        a.set_ylabel("per-channel r, LFP envelope")
        lo_ = a.get_ylim()[0]
        for i, b in enumerate(BANDS):                               # a whisker cut by the axis: its end is written
            for j, g in enumerate(GROUPS):
                q10 = np.nanpercentile(cr[f"{g}/codec_125/{b}"], 10)
                if q10 < lo_:
                    a.text(i + (j - 1) * 0.26, lo_ + 0.002, f"↓{q10:.2f}", fontsize=4.8, ha="center", va="bottom",
                           color=C["x_hat"], bbox=dict(fc="white", ec="none", pad=0.3))
        a.legend(handles=[Patch(fc="white", ec=C["x_hat"], label="codec"),
                          Line2D([], [], color=C["pca"], label="PCA (median)")] + group_handles(C["x_hat"]),
                 loc="lower left", fontsize=5.2, bbox_to_anchor=(0, 0.06))
        ygrid(a)
    letter(a, "a", "the LFP is kept in every band", dx=-0.2)
    # b · MUA per channel
    a = fig.add_subplot(gs[0, 1])
    if cr is None:
        pending(a, "channel_r")
    else:
        for j, g in enumerate(GROUPS):
            for k, src, off in (("x_hat", "codec_125", -0.17), ("pca", "pca_F240", 0.17)):
                v = cr[f"{g}/{src}/mua"].ravel()
                v = v[np.isfinite(v)]
                q = np.percentile(v, [10, 25, 50, 75, 90])
                xpos = j + off
                a.scatter(xpos + np.random.default_rng(j).uniform(-0.08, 0.08, len(v)), v, s=0.8, color=C[k],
                          alpha=0.15, lw=0)
                a.plot([xpos] * 2, [q[0], q[4]], color=C[k], lw=0.5)
                a.add_patch(Rectangle((xpos - 0.09, q[1]), 0.18, q[3] - q[1], fc="white", ec=C[k], lw=0.6,
                                      ls=LS[k]))
                a.plot(xpos, q[2], marker=GMARK[g], color=C[k], ms=2.8)
        a.set_xticks(range(3), [GSHORT[g] for g in GROUPS])
        a.set_ylim(-0.05, 1.03)
        a.set_ylabel("per-channel r, MUA")
        a.legend(handles=[Patch(fc="white", ec=C["x_hat"], label="codec"),
                          Patch(fc="white", ec=C["pca"], ls="--", label="PCA")], loc="lower left", fontsize=5.5)
        ygrid(a)
    letter(a, "b", "MUA depends on the recording")
    # c · example MUA
    a = fig.add_subplot(gs[0, 2])
    if ex is None:
        pending(a, "examples")
    else:
        mx, mh = ex["mua_x"], ex["mua_x_hat"]
        z = lambda v: (v - mx.mean()) / mx.std()
        t = (np.arange(len(mx)) - len(mx) // 2) / 1000
        a.plot(t, z(mx), color=C["grey"], lw=0.5, label="original")
        a.plot(t, z(mh), color=C["x_hat"], lw=0.5, label="codec")
        a.set_xlim(-1, 1)
        a.set_xlabel("time (s)")
        a.set_ylabel("MUA (z)")
        a.legend(loc="upper left", fontsize=5.5, ncol=2)
        a.set_title(f"one P4 electrode, r = {float(ex['mua_r']):.2f}", loc="left", fontsize=6, pad=3)
    letter(a, "c")
    # d · validation -> test
    sub = gs[1, :2].subgridspec(1, 4, wspace=0.55)
    mets = [("retention_x_hat", "retention $\\hat{x}$", None), ("retention_y_hat", "retention $\\hat{y}$", None),
            ("f1_y_hat", "F1 $\\hat{y}$", ("detect_x", "f1")), ("sorting_acc", "sorting $\\hat{x}$",
                                                                  ("sorting_x", "acc_median"))]
    p4_par = all(arm(S10, "p4_unseen", "s8-ref", m)[0] >= min(arm(S10, g, "s8-ref", m)[0] for g in VAL)
                 for m, _, _ in mets)
    d_title = ("the unseen patient scores as well as the later days" if p4_par else
               "validation (model selection) against test")
    for n_, (m, title, ceil) in enumerate(mets):
        a = fig.add_subplot(sub[n_])
        for i, g in enumerate(GROUPS):
            mu_t, sd_t = arm(S10, g, "s8-ref", m)
            if g in VAL:
                mu_v, sd_v = arm(S9, VAL[g], "s8-ref", m)
                a.errorbar(i - 0.15, mu_v, sd_v, marker=GMARK[g], color=C["grey"], mfc="white", ms=3, capsize=1, lw=0.6)
                a.plot([i - 0.15, i + 0.15], [mu_v, mu_t], color=C["grey"], lw=0.5)
            a.errorbar(i + 0.15, mu_t, sd_t, marker=GMARK[g], color=C["x_hat"], ms=3, capsize=1, lw=0.6)
            if ceil:
                c = get(ev("s8-ref-s1", g), ceil)
                a.plot([i - 0.3, i + 0.3], [c, c], color=INK, lw=0.6)
        a.set_xticks(range(3), [GSHORT[g] for g in GROUPS])
        a.set_xlim(-0.5, 2.5)
        a.set_title(title, loc="left", fontsize=6, pad=3)
        a.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ygrid(a)
        if n_ == 0:
            letter(a, "d", None, dx=-0.75, dy=1.17)
            a.text(-0.5, 1.17, d_title, transform=a.transAxes, fontsize=6.5, va="bottom")
            a.legend(handles=[Line2D([], [], marker="o", ls="", color=C["grey"], mfc="white", ms=3, label="validation"),
                              Line2D([], [], marker="o", ls="", color=C["x_hat"], ms=3, label="test"),
                              Line2D([], [], color=INK, lw=0.6, label="original $x$")],
                      loc="lower left", fontsize=5.5, bbox_to_anchor=(2.9, 1.12), ncol=3)
    # e · sorting per unit
    a = fig.add_subplot(gs[1, 2])
    if so is None:
        pending(a, "sorting")
    else:
        per = smeta["per_unit"]
        for g in GROUPS:
            x_ = np.array([v if v is not None else np.nan for v in per[g]["x"]], float)
            y_ = np.array([v if v is not None else np.nan for v in per[g]["x_hat"]], float)
            a.scatter(x_, y_, s=3, marker=GMARK[g], color=C["x_hat"], alpha=0.35, lw=0)
        a.plot([0, 1], [0, 1], color=INK, lw=0.5, ls="--")
        a.set_xlim(0, 1), a.set_ylim(0, 1), a.set_aspect("equal")
        a.set_xlabel("sorting accuracy, original")
        a.set_ylabel("sorting accuracy, codec $\\hat{x}$")
        fr = on_or_above(per)
        a.text(0.97, 0.04, "on or above the diagonal:\n" + ", ".join(f"{GSHORT[g]} {f * 100:.0f} %" for g, f in fr.items()),
               transform=a.transAxes, fontsize=5.3, va="bottom", ha="right", bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
    letter(a, "e", ("most units sort as well or better" if so is not None and min(fr.values()) >= 0.5 else
                    "sorting accuracy per unit"), dx=-0.3)
    save(fig, "Fig4_features", """
**Figure 4 | Besides spikes, the codec keeps the LFP and sorts units at least as well as the original; on a patient never seen in training the results are as good as on the training patients' later days.**
**a**, Correlation between the band envelopes (Hilbert amplitude) of the original and the codec output, per channel and
recording, for each LFP band and test group (box: quartiles, line: 10–90 %; a line cut by the axis has its end written); orange: PCA median. Linear compression keeps the slow LFP almost exactly, so
the LFP is a check that nothing is broken, not where the codec gains. **b**, Per-channel correlation of the multi-unit
activity (MUA: 500–5000 Hz rectified envelope, 1 kHz), codec vs PCA. The codec removes background activity above about
1.5 kHz, so MUA agreement depends on how much of a recording's MUA is that background. **c**, Two seconds of MUA on one
P4 electrode (r over the whole recording). **d**, Validation (earlier late days, used to select the model; grey open) against test (blue) for four
headline metrics, mean ± SD over 5 seeds; black: the original recording. **e**, Offline sorting accuracy per NIX Good
unit (SpikeForest accuracy after Hungarian matching), sorting the codec output against sorting the original with the same
sorter; every test group, 125 Hz seed 1.""")


# ------------------------------------------------------------------ Fig. 5 · real-time streaming

def fig5():
    d, meta = npz("streaming")
    fig = plt.figure(figsize=(FULL, 100 * MM))
    gs = fig.add_gridspec(2, 2, hspace=0.95, wspace=0.32)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    if d is None:
        for a, l in zip(axes, "abcd"):
            pending(a, "streaming")
            letter(a, l)
        return save(fig, "Fig5_streaming", "**Figure 5 | Real-time streaming** (awaiting data).")
    R = meta["rates"]
    has_gpu = len(d["125/ms_gpu"]) > 0
    dev = meta.get("gpu", "GPU").replace("NVIDIA ", "") if has_gpu else "CPU"
    v = d["125/ms_gpu"] if has_gpu else d["125/ms_cpu"]
    p50, p99 = np.percentile(v, 50), np.percentile(v, 99)
    fr = R["125"]["budget_ms"]
    # a · one frame
    a = axes[0]
    for k in range(3):
        s = k * fr
        a.add_patch(Rectangle((s, 2.0), fr, 0.7, fc=C["x_hat_box"], ec=C["x_hat"], lw=0.5))
        a.text(s + fr / 2, 2.35, f"frame {k + 1}", ha="center", va="center", fontsize=5.5)
        a.add_patch(Rectangle((s + fr, 1.05), p50, 0.6, fc=C["x_hat"], ec="none"))
        a.add_patch(Rectangle((s + fr + p50, 1.05), max(p99 - p50, 0.05), 0.6, fc=C["x_hat"], alpha=0.35, ec="none"))
        a.add_patch(Rectangle((s + fr + p99, 0.1), fr, 0.6, fc=C["y_hat_tint"], ec=C["y_hat"], lw=0.5))
        a.text(s + fr + p99 + fr / 2, 0.4, f"output {k + 1}", ha="center", va="center", fontsize=5.5)
    for y, x0, lab in ((3.2, 0, f"oldest sample of a frame: {fr + p99:.1f} ms to output"),
                       (3.85, fr, f"newest sample: {p99:.1f} ms")):
        a.annotate("", (fr + p99, y), (x0, y), arrowprops=dict(arrowstyle="<->", lw=0.6, shrinkA=0, shrinkB=0))
        a.text(x0, y + 0.12, lab, ha="left", va="bottom", fontsize=5.8)
    a.set_xlim(-0.3, 4 * fr + p99 + 0.5), a.set_ylim(0, 4.6)
    a.set_yticks([0.4, 1.35, 2.35], ["output", f"compute ({dev})", "input (240 samples)"])
    a.tick_params(axis="y", length=0)
    a.spines["left"].set_visible(False)
    a.set_xlabel("time (ms)")
    letter(a, "a", f"each frame is out {p99:.1f} ms after its last sample", dx=-0.28)
    # b · fast enough
    a = axes[1]
    for r in RATES:
        for devk, ls in (("gpu", "-"), ("cpu", "--")):
            vv = np.sort(d[f"{r}/ms_{devk}"])
            if len(vv):
                a.plot(vv, np.arange(1, len(vv) + 1) / len(vv), color=C[r], ls=ls, lw=1.0)
        a.axvline(R[r]["budget_ms"], color=C[r], lw=0.6, ls=":")
    a.set_xscale("log")
    a.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([1, 2, 5, 10, 20, 50, 100, 200]))
    a.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a.set_ylim(0, 1)
    a.set_xlabel("compute per frame, 96 electrodes (ms)")
    a.set_ylabel("share of frames")
    rate_h = [Line2D([], [], color=C[r], label=f"{r} Hz") for r in RATES]
    style_h = [Line2D([], [], color=INK, label=dev if has_gpu else "GPU"),
               Line2D([], [], color=INK, ls="--", label=f"CPU, {meta['torch_threads']} threads"),
               Line2D([], [], color=INK, ls=":", label="budget = frame length")]
    a.legend(handles=[h for pair in zip(rate_h, style_h) for h in pair], loc="upper center",
             bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=5.3, columnspacing=1.2)
    ygrid(a)
    ok = [r for r in RATES if len(d[f"{r}/ms_gpu"]) and np.percentile(d[f"{r}/ms_gpu"], 99) < R[r]["budget_ms"]]
    letter(a, "b", "real time on a GPU at every rate" if len(ok) == len(RATES) else
           f"real time on a GPU at {', '.join(ok) or 'no'} Hz", dx=-0.2)
    # c · never looks ahead
    a = axes[2]
    r = "125"
    t0, f = R[r]["t0"], R[r]["frame_samples"]
    ms = lambda i: (np.asarray(i) - t0) / HZ * 1e3
    win = np.arange(max(t0 - 3 * f, 0), min(t0 + 3 * f, len(d[f"{r}/pert_y_a"])))
    a.axvspan(ms(R[r]["frame_start_of_t0"]), ms(win[-1]), color="#FBEDE5", lw=0, zorder=0)
    for b in range((win[0] // f) * f, win[-1] + 1, f):
        a.axvline(ms(b), color=C["grey"], lw=0.4, ls=":")
    a.plot(ms(win), d[f"{r}/pert_y_a"][win], color=C["x_hat"], lw=0.9, label="original input")
    a.plot(ms(win), d[f"{r}/pert_y_b"][win], color=C["accent"], lw=0.8, ls="--", label="input changed at 0 ms")
    a.set_xlim(ms(win[0]), ms(win[-1]))
    a.set_xlabel("time from the change (ms); dotted: frame edges")
    a.set_ylabel("$\\hat{x}$ (σ)")
    a.legend(loc="lower left", fontsize=5.5)
    letter(a, "c", "the output never changes before the frame that changed", dx=-0.2)
    # d · start-up
    a = axes[3]
    for r in RATES:
        e = d[f"{r}/startup_rel_err"]
        a.semilogy((np.arange(len(e)) + 0.5) * 10, np.maximum(e, 1e-7), color=C[r], lw=1.0, label=f"{r} Hz")
    a.set_xlim(0, 1000)
    a.axvline(11352 / HZ * 1e3, color=C["grey"], lw=0.5, ls="--")
    a.text(11352 / HZ * 1e3 + 15, 0.55, "receptive field\n(125 Hz)", transform=a.get_xaxis_transform(), fontsize=5.5)
    a.set_xlabel("time since a cold start (ms)")
    a.set_ylabel("difference to a running stream")
    a.legend(loc="upper right", fontsize=5.5)
    ygrid(a)
    letter(a, "d", "a cold start settles within the receptive field", dx=-0.2)
    pr = R["125"]
    gpu_txt = "; ".join(f"{r} Hz {np.percentile(d[f'{r}/ms_gpu'], 99):.1f} of {R[r]['budget_ms']:.0f} ms"
                        for r in RATES if len(d[f"{r}/ms_gpu"]))
    cpu_txt = "; ".join(f"{r} Hz {np.percentile(d[f'{r}/ms_cpu'], 99):.1f} of {R[r]['budget_ms']:.0f} ms"
                        for r in RATES)
    kept = "; ".join(f"{r} Hz {d[f'{r}/startup_kept_cold'][0] / max(d[f'{r}/startup_found'][0], 1):.3f} vs "
                     f"{d[f'{r}/startup_kept_warm'][0] / max(d[f'{r}/startup_found'][0], 1):.3f}" for r in RATES)
    save(fig, "Fig5_streaming", f"""
**Figure 5 | The codec runs in real time, never uses future input, and settles within its receptive field after a cold start.**
**a**, What happens to each 8 ms frame at 125 Hz: its 240 samples arrive, the encoder and both decoders run on all 96
electrodes in one call (dark: median {p50:.1f} ms, light: up to the 99th percentile {p99:.1f} ms, {dev}), and the outputs for
that frame are released. The oldest sample of a frame is released {fr + p99:.1f} ms after it was recorded, the newest
{p99:.1f} ms after. **b**, Compute time per frame (cumulative share of 2 000 frames on {dev}, 500 on CPU) against each rate's
budget (the frame length). 99th percentiles — {dev}: {gpu_txt}; CPU ({meta['torch_threads']} threads): {cpu_txt}.
**c**, Causality on real data (one P4 electrode): the input is replaced from 0 ms on; the outputs for the original and
the changed input are identical up to the start of the frame that contains the change (shaded from there on) (maximum difference
{pr['max_abs_before_frame']:.0e}) — the first output sample that moves lies {pr['lookback_samples'] / HZ * 1e3:.1f} ms
before the change, inside the same frame. Streaming frame by frame equals one offline pass (maximum difference
{pr['stream_vs_offline_max_abs']:.0e}). **d**, After a cold start (empty state), the difference to a stream that was
already running (RMS, in units of the signal SD) falls to floating-point level once the receptive field is filled. Spikes kept in the first second, cold
vs running: {kept}.""")


# ------------------------------------------------------------------ supplementary

def s1():
    d, _ = npz("psd")
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 50 * MM), sharey=True, gridspec_kw=dict(wspace=0.12))
    for j, g in enumerate(GROUPS):
        a = axes[j]
        if d is None:
            pending(a, "psd")
            continue
        f = d["freq"]
        k = (f >= 1) & (f <= 15000)
        a.axvspan(500, 7500, color=C["y_hat_tint"], lw=0)
        a.loglog(f[k], d[f"{g}/x"][k], color=INK, lw=1.0, label="original")
        for r in RATES:
            a.loglog(f[k], d[f"{g}/codec_{r}"][k], color=C[r], lw=0.8, label=f"codec {r} Hz")
        a.loglog(f[k], d[f"{g}/pca_F240"][k], color=C["pca"], lw=0.8, ls="--", label="PCA (125 Hz budget)")
        a.set_xlabel("frequency (Hz)")
        a.set_title(GLABEL[g], loc="left", fontsize=6.5)
        letter(a, "abc"[j])
    axes[0].set_ylabel("power (σ² / Hz)")
    if d is not None:
        axes[0].legend(loc="lower left", fontsize=5.3)
    save(fig, "S1_spectrum", """
**Supplementary Figure 1 | The codec keeps the spectrum up to the spike band and drops the background above it.**
Welch power spectral density (8 192-sample windows, median over channels and recordings, first 30 s) of the original,
the codec at each token rate and PCA at the 125 Hz budget, per test group. Shaded: the spike band (500–7500 Hz). The
codec keeps the LFP and the spike band and removes background power above about 1.5 kHz (125 Hz) and 0.8 kHz (62.5 Hz),
which explains the MUA results of Fig. 4b and the detection gain of Fig. 2e.""")


def s2():
    """F8 layout: the units are as separable on the codec output as on the original."""
    so, smeta = npz("sorting")
    fig = plt.figure(figsize=(FULL, 62 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.32)
    if so is None:
        for j, l in enumerate("abc"):
            a = fig.add_subplot(gs[j])
            pending(a, "sorting")
            letter(a, l)
        return save(fig, "S2_sorting", "**Supplementary Figure 2 | Spike sorting on the codec output** (awaiting data).")
    units = sorted({int(u) for u in np.unique(so["x/nix_unit"]) if u >= 0})
    col = {u: UNIT[i % len(UNIT)] for i, u in enumerate(units)}
    for j, (k, title) in enumerate((("x", "snippets of the original"), ("x_hat", "snippets of the codec output"))):
        a = fig.add_subplot(gs[j])
        f, nu = so[f"{k}/feat"], so[f"{k}/nix_unit"]
        a.scatter(f[nu < 0, 0], f[nu < 0, 1], s=1, color=C["light"], lw=0, rasterized=True)
        for u in units:
            a.scatter(f[nu == u, 0], f[nu == u, 1], s=1.5, color=col[u], lw=0, rasterized=True)
        a.set_xticks([]), a.set_yticks([])
        a.set_xlabel("PC 1"), a.set_ylabel("PC 2")
        letter(a, "ab"[j], title, dx=-0.08)
        if j == 0:
            a.legend(handles=[Line2D([], [], marker="o", ls="", ms=2.5, color=col[u], label=f"NIX unit {n + 1}")
                              for n, u in enumerate(units)] +
                     [Line2D([], [], marker="o", ls="", ms=2.5, color=C["light"], label="no NIX unit")],
                     loc="lower left", fontsize=5.5, handletextpad=0.2)
    a = fig.add_subplot(gs[2])
    per = smeta["per_unit"]
    for g in GROUPS:
        x_ = np.array([v if v is not None else np.nan for v in per[g]["x"]], float)
        y_ = np.array([v if v is not None else np.nan for v in per[g]["x_hat"]], float)
        a.scatter(x_, y_, s=4, marker=GMARK[g], color=C["x_hat"], alpha=0.4, lw=0, rasterized=True)
    fr = on_or_above(per)
    a.plot([0, 1], [0, 1], color=INK, lw=0.5, ls="--")
    a.set_xlim(0, 1), a.set_ylim(0, 1), a.set_aspect("equal"), a.set_anchor("N")
    a.set_xlabel("accuracy, sorting the original")
    a.set_ylabel("accuracy, sorting the codec output")
    a.legend(handles=group_handles(C["x_hat"]), loc="lower right", fontsize=5.5, handletextpad=0.2,
             title="on or above the diagonal", title_fontsize=5.5, frameon=True, framealpha=0.9, edgecolor="none",
             labels=[f"{GLABEL[g]}: {f * 100:.0f} %" for g, f in fr.items()])
    letter(a, "c", "most units sort as well or better" if min(fr.values()) >= 0.5 else "sorting accuracy per unit",
           dx=-0.25)
    ex = smeta["example"]
    save(fig, "S2_sorting", f"""
**Supplementary Figure 2 | Spike sorting on the codec output finds the same units as on the original.**
**a, b**, Every −5σ crossing of the spike band on one P4 electrode (recording {ex['session']}, row {ex['row']}) as a
61-sample snippet projected on its first two principal components, coloured by the NIX Good unit it matches (±0.5 ms;
light grey: no Good unit), for the original recording (**a**) and the codec output $\\hat x$ (**b**). **c**, Offline
sorting accuracy per NIX Good unit (SpikeForest accuracy after Hungarian matching; the same sorter on both signals),
codec output against original, for every unit of the three test groups (marker shape: group); dashed: equal accuracy.
125 Hz model, seed 1.""")


def online_window(ex, win_s=0.15):
    """S5: the example window, by a fixed rule. Among the electrodes where the online sorter holds templates for
    at least two NIX units (if any), the win_s window with the most NIX Good spikes of at least two units. Each online
    ID is drawn in the colour of the NIX unit its spikes match most often (the whole 1 s); events are (t, colour, note)."""
    def n_ids(i):
        nt, nu, ot, ol = (ex[f"{i}/{k}"] for k in ("nix_t", "nix_unit", "online_t", "online_label"))
        hit = [(int(l), int(nu[np.argmin(np.abs(nt - t))])) for t, l in zip(ot, ol)
               if l >= 0 and len(nt) and np.abs(nt - t).min() <= 15]
        return len({max({u for l_, u in hit if l_ == l}, key=[u for l_, u in hit if l_ == l].count)
                    for l in {l for l, _ in hit}})
    cand = [i for i in range(len(ex["rows"])) if n_ids(i) >= 2] or list(range(len(ex["rows"])))
    win, best = int(win_s * HZ), (-1, 0, 0)
    for i in cand:
        nt, nu = ex[f"{i}/nix_t"], ex[f"{i}/nix_unit"]
        for b in range(0, HZ - win, HZ // 100):
            k = (nt >= b) & (nt < b + win)
            score = k.sum() if len(np.unique(nu[k])) >= 2 else 0
            if score > best[0]:
                best = (score, i, b)
    _, i, b0 = best
    nt, nu = ex[f"{i}/nix_t"], ex[f"{i}/nix_unit"]
    ot, ol = ex[f"{i}/online_t"], ex[f"{i}/online_label"]
    near = [int(nu[np.argmin(np.abs(nt - t))]) if len(nt) and np.abs(nt - t).min() <= 15 else -1 for t in ot]
    to_nix = {}
    for lab in np.unique(ol[ol >= 0]):
        m = [n for n, l_ in zip(near, ol) if l_ == lab and n >= 0]
        to_nix[int(lab)] = max(set(m), key=m.count) if m else -1
    units = sorted({int(u) for u in nu}, key=lambda u: (-(nu == u).sum(), u))      # busiest unit gets the darkest shade
    col = {u: UNIT[n % len(UNIT)] for n, u in enumerate(units)}
    online, said = [], set()
    for t, lab, n in zip(ot, ol, near):
        inside = b0 + 15 <= t < b0 + win - 30
        if lab < 0:
            online.append((int(t), C["grey"], "unsorted" if inside and "u" not in said else None))
            said |= {"u"} if inside else set()
        else:
            u = to_nix[int(lab)]
            wrong = inside and n >= 0 and u != n
            online.append((int(t), col.get(u, C["grey"]), "wrong unit" if wrong and "w" not in said else None))
            said |= {"w"} if wrong else set()
    return dict(i=i, b0=b0, win=win, sig=float(ex["sigma_bp"][i]), units=units, col=col, cand=cand,
                ref=[(int(t), col[int(u)], None) for t, u in zip(nt, nu)], online=online)


def online_row(a, v, events, w, title, base=None):
    """One row of S5 a: the trace in `base` (light grey by default; ŷ in its own colour), each event's spike in its
    colour, a note (arrow) where one is given."""
    b0, win = w["b0"], w["win"]
    tms = (np.arange(b0, b0 + win) - b0) / HZ * 1e3
    a.plot(tms, v[b0:b0 + win], color=base or C["trace_bg"], lw=0.4)
    for t, c, mark in events:
        if b0 + 15 <= t < b0 + win - 30:
            s_ = slice(t - 15, t + 30)
            a.plot((np.arange(s_.start, s_.stop) - b0) / HZ * 1e3, v[s_], color=c, lw=0.9)
            if mark:
                f = (t - b0) / win                                     # notes near an edge lean inward
                a.annotate(mark, ((t - b0) / HZ * 1e3, v[s_].max()), (0, 9), textcoords="offset points",
                           ha="right" if f > 0.85 else "left" if f < 0.15 else "center",
                           fontsize=5.5, color=MUTED,
                           arrowprops=dict(arrowstyle="-|>", lw=0.5, color=MUTED, shrinkA=0, shrinkB=1,
                                           mutation_scale=10))
    a.set_yticks([]), a.spines["left"].set_visible(False)
    a.set_xlim(0, win / HZ * 1e3)
    lo, hi = a.get_ylim()
    a.set_ylim(lo, hi + 0.45 * (hi - lo))      # room for the arrows and the key
    a.set_title(title, loc="left", fontsize=6.5)


def s5():
    """F12 layout: the online sorter on the spike-decoder output, next to the curated reference."""
    ex, emeta = npz("examples")
    so, smeta = npz("sorting")
    fig = plt.figure(figsize=(FULL, 70 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[3.2, 1], height_ratios=[1, 1], hspace=0.55, wspace=0.25)
    if ex is None or so is None:
        for sl, l in ((gs[:, 0], "a"), (gs[:, 1], "b")):
            a = fig.add_subplot(sl)
            pending(a, "examples / sorting")
            letter(a, l)
        return save(fig, "S5_online", "**Supplementary Figure 5 | Online unit IDs** (awaiting data).")
    w = online_window(ex)
    i, sig, col, units, cand = w["i"], w["sig"], w["col"], w["units"], w["cand"]
    a = fig.add_subplot(gs[0, 0])
    online_row(a, ex[f"{i}/bp_x"] / sig, w["ref"], w, "reference: NIX Good units on the original")
    a.tick_params(labelbottom=False)
    letter(a, "a", None, dx=-0.03)
    top = a
    a = fig.add_subplot(gs[1, 0], sharex=top)
    online_row(a, ex[f"{i}/y_hat"] / sig, w["online"], w, "spike decoder $\\hat{y}$: online unit IDs",
               base=C["y_hat_line"])
    a.set_xlabel("time (ms)")
    top.legend(handles=[Line2D([], [], color=col[u], lw=1.2, label=f"unit {n + 1}") for n, u in enumerate(units)] +
               [Line2D([], [], color=C["grey"], lw=1.2, label="unsorted")], loc="lower right",
               ncol=len(units) + 1, fontsize=5.5, bbox_to_anchor=(1.0, 1.0), handlelength=1.2, columnspacing=0.8)
    a = fig.add_subplot(gs[:, 1])
    per = smeta["per_unit"]
    for k, g in enumerate(GROUPS):
        v = np.array([x for x in per[g]["y_hat_online"] if x is not None], float)
        if not len(v):
            continue
        q = np.percentile(v, [25, 50, 75])
        a.scatter(k + np.random.default_rng(k).uniform(-0.15, 0.15, len(v)), v, s=1.5, color=C["y_hat"], alpha=0.35,
                  lw=0, rasterized=True)
        a.add_patch(Rectangle((k - 0.14, q[0]), 0.28, q[2] - q[0], fc="none", ec=C["y_hat"], lw=0.7))
        a.plot(k, q[1], marker=GMARK[g], color=C["y_hat"], ms=3.5)
        a.plot([k - 0.28, k + 0.28], [get(ev("s8-ref-s1", g), ("sorting_x", "acc_median"))] * 2, color=INK, lw=0.8)
    a.set_xticks(range(3), [GSHORT[g] for g in GROUPS])
    a.set_xlim(-0.5, 2.5), a.set_ylim(0, 1)
    a.set_ylabel("online accuracy per unit")
    a.legend(handles=[Line2D([], [], color=INK, lw=0.8, label="offline, original (median)")], loc="lower left",
             fontsize=5.3, handlelength=1.2, frameon=True, framealpha=0.9, edgecolor="none")
    ygrid(a)
    letter(a, "b", "per unit, every test group", dx=-0.35)
    save(fig, "S5_online", f"""
**Supplementary Figure 5 | Unit IDs can be given online, at the receiver, from the spike decoder output alone.**
**a**, 150 ms of one P4 electrode (recording {emeta['session']}, row {int(ex['rows'][i])}; chosen by rule: of the
{len(ex['rows'])} example electrodes, {len(cand)} where the online sorter holds templates for at least two NIX units, and there
the window with the most NIX Good spikes of at least two units). Top: the spike band of the original with the NIX Good spikes coloured
by unit (the offline, curated reference). Bottom: the spike decoder output $\\hat y$ with every −5σ crossing coloured by
the online sorter's ID (nearest template; templates from the first 60 s), each ID drawn in the colour of the NIX unit
its spikes match most often; grey: no template close enough (unsorted). **b**, Online accuracy per NIX Good unit on
$\\hat y$ (SpikeForest accuracy; box: quartiles, marker: median) for the three test groups; black line: median offline
accuracy of sorting the original recording. 125 Hz model, seed 1.""")


def s3():
    ex, _ = npz("examples")
    fig, a = plt.subplots(figsize=(FULL, 42 * MM))
    if ex is None:
        pending(a, "examples")
        return save(fig, "S3_raster", "**Supplementary Figure 3 | Detection raster** (awaiting data).")
    good, bad = ex["0/nix_t"], ex["0/nix_bad_t"]
    rows = [("NIX Good", None), ("bp($x$)", "x"), ("bp($\\hat{x}$)", "x_hat"), ("$\\hat{y}$", "y_hat")]
    for i, (_label, k) in enumerate(rows):
        y = len(rows) - 1 - i
        if k is None:
            a.vlines(good / HZ, y - 0.35, y + 0.35, color=INK, lw=0.6)
            continue
        det = ex[f"0/det_{k}"]
        near = lambda ref: np.array([len(ref) and np.abs(ref - t).min() <= 15 for t in det], bool)
        hit, other = near(good), near(bad)
        a.vlines(det[hit] / HZ, y - 0.35, y + 0.35, color=C["lowrate"], lw=0.6)
        a.vlines(det[~hit & other] / HZ, y - 0.35, y + 0.35, color=C["grey"], lw=0.6)
        a.vlines(det[~hit & ~other] / HZ, y - 0.35, y + 0.35, color=C["accent"], lw=0.8)
        miss = np.array([t for t in good if not len(det) or np.abs(det - t).min() > 15])
        if len(miss):
            a.vlines(miss / HZ, y - 0.35, y + 0.35, color="#BBBBBB", lw=0.5, ls=(0, (1, 1)))
    a.set_yticks(range(4), [r[0] for r in rows][::-1])
    a.set_xlim(0, 1), a.set_xlabel("time (s)")
    a.spines["left"].set_visible(False), a.tick_params(axis="y", length=0)
    a.legend(handles=[Line2D([], [], color=C["lowrate"], label="hit"), Line2D([], [], color="#BBBBBB", ls=":", label="miss"),
                      Line2D([], [], color=C["accent"], label="false alarm"), Line2D([], [], color=C["grey"], label="NIX Bad unit")],
             loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=4, fontsize=5.5)
    save(fig, "S3_raster", """
**Supplementary Figure 3 | Detection on one electrode.**
One second of one P4 electrode: NIX Good spike times and the −5σ crossings on the original, the codec reconstruction and
the spike decoder output. Hit: within ±0.5 ms of a Good spike; NIX Bad: matches only a spike of a Bad unit (not counted as
false); false alarm: matches no NIX spike; miss: a Good spike with no crossing.""")


def s4():
    wf, _ = npz("waveforms")
    fig = plt.figure(figsize=(FULL, 70 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.3)
    a = fig.add_subplot(gs[1])
    for g in GROUPS:
        u = ev("s8-ref-s1", g)["units"]
        a.scatter([v["amp_sigma"] for v in u], [v["wave_r"] for v in u], s=2.5, marker=GMARK[g], color=C["x_hat"],
                  alpha=0.35, lw=0)
    a.set_xscale("log")
    a.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    a.set_xlabel("unit SNR (peak-to-peak / σ)")
    a.set_ylabel("waveform r (median per unit)")
    a.legend(handles=group_handles(C["x_hat"]), loc="lower right", fontsize=5.5)
    ygrid(a)
    letter(a, "b", dx=-0.25)
    sub = gs[0].subgridspec(3, 4, wspace=0.15, hspace=0.45)
    if wf is None:
        aa = fig.add_subplot(sub[:, :])
        pending(aa, "waveforms")
    else:
        snr = wf["snr"]
        o = np.argsort(snr)
        pick = o[(np.linspace(0.02, 0.99, 12) * (len(o) - 1)).astype(int)][::-1]
        t = (np.arange(61) - 21) / 30
        for n_, u in enumerate(pick):
            aa = fig.add_subplot(sub[n_ // 4, n_ % 4])
            aa.fill_between(t, wf["bp_x_mean"][u] - wf["bp_x_sd"][u], wf["bp_x_mean"][u] + wf["bp_x_sd"][u],
                            color=INK, alpha=0.1, lw=0)
            aa.plot(t, wf["bp_x_mean"][u], color=INK, lw=0.8)
            aa.plot(t, wf["bp_x_hat_mean"][u], color=C["x_hat"], lw=0.8, ls="--")
            aa.plot(t, wf["y_hat_mean"][u], color=C["y_hat"], lw=0.8, ls=":")
            aa.set_title(f"{snr[u]:.0f}σ · r {wf['wave_r'][u]:.2f}", fontsize=5.3, loc="center", pad=1.5)
            aa.tick_params(labelsize=5)
            if n_ % 4:
                aa.set_yticks([])
            if n_ // 4 < 2:
                aa.set_xticks([])
            if n_ == 0:
                letter(aa, "a", dx=-0.5)
    save(fig, "S4_waveforms", """
**Supplementary Figure 4 | Waveforms of twelve P4 units and waveform correlation for every unit.**
**a**, Mean spike-band waveforms (σ units) on the original (black, ± SD), the codec reconstruction (dashed) and the spike
decoder (dotted) for twelve P4 units spanning the SNR range. **b**, Median per-spike correlation of bp($\\hat x$) with bp($x$)
for every NIX Good unit of the three test groups against the unit's SNR (125 Hz, seed 1). NIX templates are not shown:
they are whitened and on another scale.""")


# ------------------------------------------------------------------ checks and main

def check():
    """Every plotted headline equals s10.json; the identity baseline keeps every spike."""
    paths = {"retention_x_hat": ("retention_x_hat", "retention"), "retention_y_hat": ("retention_y_hat", "retention"),
             "f1_y_hat": ("detect_y_hat", "f1"), "sorting_acc": ("sorting_x_hat", "acc_median")}
    for g in GROUPS:
        assert base("identity", 240, g)["retention_x_hat"]["retention"] == 1.0, g
        for m, p in paths.items():
            assert abs(seeds(g, p).mean() - arm(S10, g, "s8-ref", m)[0]) < 5e-4, (g, m)
    print("checks passed: identity = 1.0, eval JSONs = s10.json")


FEATURE_LABEL = {"spike": "NIX Good spike", "crossing": "−5σ crossing", "artifact": "artifact", "position": "spike time in frame",
                 "trough": "spike depth", "spike_rms": "spike-band RMS", "mua": "MUA", "voltage": "slow voltage",
                 **{f"{k}_signed": f"LFP {v} signal" for k, v in BANDS.items()}, **{k: f"LFP {v} log power" for k, v in BANDS.items()}}


def s_latent(g="p4_unseen"):
    """What the 32 numbers hold: a map to look at, a table that carries the numbers. Read from `figure_data latent`."""
    d, m = npz("latent")
    if d is None or g not in m["groups"]:
        print("S_latent: no data yet (python -m scripts.figure_data latent)")
        return
    meta = m["groups"][g]
    names, rho = meta["features"], d[f"{g}/rho"]
    fig = plt.figure(figsize=(FULL, 170 * MM))
    outer = fig.add_gridspec(3, 2, width_ratios=[1, 1], height_ratios=[1.05, 1.5, 0.8], wspace=0.34, hspace=0.42)
    maps = outer[0, :].subgridspec(2, 3, wspace=0.05, hspace=0.16)
    xy, spk = d[f"{g}/map"], d[f"{g}/map_spike"]

    def blank(a, title):
        a.set_xticks([]), a.set_yticks([])
        for s in a.spines.values():
            s.set_linewidth(0.4)
        a.set_title(title, loc="left", fontsize=5.8, pad=1.5)

    # a: one map, coloured four ways; points in a fixed random order, so neither class hides the other
    a = fig.add_subplot(maps[0, 0])
    order = np.random.default_rng(0).permutation(len(xy))
    a.scatter(*xy[order].T, s=0.5, lw=0, c=np.where(spk[order], C["y_hat"], C["trace_bg"]), rasterized=True)
    blank(a, "spike in the token")
    a.legend(handles=[Line2D([], [], ls="", marker="o", ms=2.2, color=C["y_hat"], label="spike"),
                      Line2D([], [], ls="", marker="o", ms=2.2, color=C["trace_bg"], label="none")],
             loc="lower right", fontsize=4.8, handletextpad=0.1, borderpad=0.2, frameon=True)
    letter(a, "a", dx=-0.12, dy=1.05)
    for pos, key, title in (((0, 1), "position", "spike time in the 8 ms frame"), ((0, 2), "trough", "spike depth"),
                            ((1, 0), "voltage", "slow voltage"), ((1, 1), "mua", "MUA")):
        a = fig.add_subplot(maps[pos])
        v = d[f"{g}/map_{key}"][order]
        k = np.isfinite(v)                                  # position and depth exist only on spike tokens
        lo, hi = np.percentile(v[k], [1, 99])
        a.scatter(*xy[order][~k].T, s=0.5, lw=0, c=C["light"], rasterized=True)
        a.scatter(*xy[order][k].T, s=0.5, lw=0, c=v[k], cmap="cividis", vmin=lo, vmax=hi, rasterized=True)
        blank(a, title + " (dark low, yellow high)")
    a = fig.add_subplot(maps[1, 2])
    if f"{g}/unit_map" in d:
        ux, lab = d[f"{g}/unit_map"], d[f"{g}/unit_label"]
        for u in range(lab.max() + 1):
            a.scatter(*ux[lab == u].T, s=0.9, lw=0, color=UNIT[u % len(UNIT)], marker="os^"[u % 3], rasterized=True)
        blank(a, f"units on one electrode ({lab.max() + 1}), own map")
    else:
        pending(a, "no electrode with 2 units")

    # b: per dimension |rho|, dimensions by variance; whole-token readouts beside it, as numbers
    gs = outer[1, :].subgridspec(1, 3, width_ratios=[32, 3.4, 3.4], wspace=0.03)
    a = fig.add_subplot(gs[0])
    dims = np.argsort(-d[f"{g}/z_var"])
    im = a.imshow(rho[:, dims], aspect="auto", cmap="cividis", vmin=0, vmax=max(0.3, float(np.nanmax(rho))), interpolation="nearest")
    a.set_yticks(range(len(names)), [FEATURE_LABEL.get(n, n) for n in names], fontsize=5.4)
    a.set_xticks([0, 7, 15, 23, 31], [1, 8, 16, 24, 32], fontsize=5.4)
    a.set_xlabel("token dimension, largest variance first", fontsize=5.8)
    a.tick_params(length=1.5)
    letter(a, "b", dx=-0.11, dy=1.03)
    a.set_title("|Spearman ρ|, each dimension alone", loc="left", fontsize=5.8, pad=1.5)
    cb = fig.colorbar(im, cax=a.inset_axes([0.68, 1.03, 0.28, 0.025]), orientation="horizontal")   # keeps a's rows level with the numbers
    cb.ax.tick_params(labelsize=4.8, length=1.2, pad=0.5, top=True, bottom=False, labeltop=True, labelbottom=False)
    for j, (key, title) in enumerate((("linear", "all 32\nlinear"), ("nonlinear", "all 32\nnonlinear"))):
        a = fig.add_subplot(gs[j + 1])
        a.set_xticks([]), a.set_yticks([])
        a.set_xlim(0, 1), a.set_ylim(len(names) - 0.5, -0.5)
        for s in a.spines.values():
            s.set_visible(False)
        a.set_title(title, fontsize=5.4, pad=1.5)
        for i, (v, b) in enumerate(zip(d[f"{g}/{key}"], d[f"{g}/binary"])):
            txt = "—" if not np.isfinite(v) else f"{v:.2f}"
            a.text(0.5, i, txt, ha="center", va="center", fontsize=5.4, style="italic" if b else "normal")

    # c: how many dimensions are used
    a = fig.add_subplot(outer[2, 0])
    val = d[f"{g}/eigvals"]
    a.plot(np.arange(1, len(val) + 1), np.cumsum(val) / val.sum(), color=C["x_hat"], marker="o", ms=1.8, lw=0.8)
    a.set_xlabel("principal components of the token")
    a.set_ylabel("variance explained (cumulative)")
    a.set_ylim(0, 1.02), a.set_xlim(0, len(val) + 1)
    ygrid(a)
    a.text(0.97, 0.08, f"participation ratio {meta['participation_ratio']:.1f} of {len(val)}", transform=a.transAxes, ha="right", fontsize=5.6)
    letter(a, "c", dx=-0.2, dy=1.08)

    # d: spike-triggered mean token
    a = fig.add_subplot(outer[2, 1])
    tr, lags = d[f"{g}/trig"], np.asarray(meta["lags"])
    ms = lags * 8
    for pc, ls in zip(range(3), ("-", "--", ":")):
        a.plot(ms, tr[0, :, pc], color=C["y_hat"], lw=1.1, ls=ls, label=f"PC{pc + 1}, all spikes")
    for u in range(1, len(tr)):
        a.plot(ms, tr[u, :, 0], color=UNIT[(u - 1) % len(UNIT)], lw=0.6, marker="os^"[(u - 1) % 3], ms=1.5, label=f"PC1, unit {u} (n = {meta['n_trig'][u]})")
    a.axvline(0, color=C["grey"], lw=0.5)
    a.axhline(0, color=C["grey"], lw=0.4)
    a.set_xlabel("token relative to the token holding the spike (ms)")
    a.set_ylabel("mean token, PC score (z-scored)")
    a.legend(fontsize=4.8, ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5), handlelength=2.0, frameon=False)
    letter(a, "d", dx=-0.2, dy=1.08)

    n_rec, n_all = len(meta["recordings"]), meta["n_recordings_in_group"]
    agree = meta["knn_agreement"]["seed0"]
    draft = "" if n_rec == n_all else f"**Draft: {n_rec} of the {n_all} recordings of this group.** "
    save(fig, "S_latent", f"""
**Supplementary Figure | What the 32 numbers of a token hold.** {draft}125 Hz model, seed 1, {GLABEL[g]},
{m['seconds']} s per recording, {meta['tokens']:,} tokens. Dimensions are not comparable between seeds.
**a**, One t-SNE of the z-scored tokens, coloured six ways (grey: no spike, so no value), {agree['n']:,} points: {agree['n'] // 2:,} tokens with a NIX Good spike and as many without
(spikes are {meta['spike_rate'] * 100:.1f} % of tokens, so they are oversampled to be visible), artifacts left out. Distances between
groups of points on a t-SNE map cannot be read quantitatively. Neighbours sharing the spike label (15 nearest, chance 0.50):
{agree['map']:.2f} on the map, {agree['tokens_32d']:.2f} in the 32 dimensions, {agree['pc12']:.2f} on the first two PCs. Last panel: spike tokens of the one
electrode with the most units, in its own map, because unit IDs mean nothing across electrodes.
**b**, Each row is a feature of the token's own 8 ms frame: slow voltage is the 1 kHz signal, an LFP signal is the signed band-passed voltage (zero-phase, so it also sees ahead, which a causal token cannot), and MUA and LFP power are log power. A single token holds the slow voltage; separating bands needs several tokens, as the decoder reads them. Left: |Spearman ρ| of each dimension alone. Right:
every dimension together, fitted on the first half of each recording and scored on the second: R², or AUC in italics for yes/no features
(0.50 = chance). Linear: ridge or logistic regression. Nonlinear: gradient-boosted trees. "—": not enough data of both classes.
**c**, PCA of the z-scored tokens. **d**, Mean token around a spike, on the first PCs of panel c; lag 0 is the token whose frame holds the spike.
Features computed with σ from bp(x); NIX Good times only.""")


FIGS = {"Fig1": fig1, "Fig2": fig2, "Fig3": fig3, "Fig4": fig4, "Fig5": fig5, "S1": s1, "S2": s2, "S3": s3, "S4": s4, "S5": s5, "S_latent": s_latent}


def main():
    check()
    for k in (sys.argv[1:] or FIGS):
        FIGS[k]()
    (OUT / "captions" / "ALL.md").write_text("\n\n".join(
        (OUT / "captions" / f"{p.stem}.md").read_text(encoding="utf-8") for p in sorted((OUT / "captions").glob("*.md"))
        if p.stem != "ALL"), encoding="utf-8")


if __name__ == "__main__":
    main()
