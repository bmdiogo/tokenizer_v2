"""Training-time monitoring figures. Each function returns ONE matplotlib figure, logged under its
own W&B key (plots/<name>), always on the same fixed validation segment so steps are comparable.

`seg` is the monitoring segment: x, x_hat, y_hat [C, T] normalised, art [C, T], ref {row: times},
sigma_bp [C], ch (the electrode with the most Good spikes), lfp_r {band: [C]}, mua_r [C].
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import signal as sps  # noqa: E402

from utils.config import HZ, env  # noqa: E402
from utils.dsp import LFP_BANDS, WF_POST, WF_PRE, bp, down_1k, lfp_band, mua  # noqa: E402
from utils.metrics import crossings  # noqa: E402

X, XH, Y = "0.15", "#d1495b", "#2a78d6"          # x (black), x̂ (red), ŷ / y (blue)


def _ms(n, t0=0):
    return (np.arange(n) + t0) / HZ * 1e3


def traces(seg):
    """x vs x̂ at 1 s, 100 ms and 5 ms, and the residual x̂ - x."""
    c, x, xh = seg["ch"], seg["x"][seg["ch"]], seg["x_hat"][seg["ch"]]
    t_spk = seg["ref"].get(c, np.array([HZ // 2]))
    centre = int(t_spk[len(t_spk) // 2])
    fig, axes = plt.subplots(2, 3, figsize=(15, 5), sharex="col")
    for j, half in enumerate((HZ // 2, HZ // 20, HZ // 400)):
        n = min(2 * half, len(x))
        a = int(np.clip(centre - n // 2, 0, len(x) - n))           # centred on a spike, inside the segment
        sl = slice(a, a + n)
        axes[0, j].plot(_ms(n, a), x[sl], color=X, lw=0.7, label="x")
        axes[0, j].plot(_ms(n, a), xh[sl], color=XH, lw=0.7, label="x̂")
        axes[1, j].plot(_ms(n, a), xh[sl] - x[sl], color="0.4", lw=0.7)
        axes[0, j].set_title(f"{n / HZ * 1e3:g} ms")
        axes[1, j].set_xlabel("time (ms)")
    axes[0, 0].set_ylabel("normalised")
    axes[1, 0].set_ylabel("residual x̂ - x")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle(f"electrode {c}: r = {np.corrcoef(x, xh)[0, 1]:.3f}")
    return fig


def _busiest(seg, n):
    """The n electrode rows with the most Good spikes in the segment (the same rows in every plot)."""
    return sorted(seg["ref"], key=lambda r: -len(seg["ref"][r]))[:n]


def waveforms(seg, n_el=6, n_spk=8):
    """Spike shapes on the busiest electrodes, in units of each electrode's noise level.
    Left: bp(x) vs bp(x̂) (does the reconstruction keep the spike?). Right: y vs ŷ (does the spike decoder
    output the clean spike?). Thin = single Good spikes, thick = their mean, dotted = the -K·σ threshold."""
    rows = [r for r in _busiest(seg, n_el)
            if ((seg["ref"][r] >= WF_PRE) & (seg["ref"][r] + WF_POST < seg["x"].shape[1])).any()]
    if not rows:
        return None
    k = env("K_THRESHOLD", 5.0)
    ms = np.arange(-WF_PRE, WF_POST) / HZ * 1e3
    fig, axes = plt.subplots(len(rows), 2, figsize=(10, 2.1 * len(rows) + 1.2), sharex=True, squeeze=False)
    for i, r in enumerate(rows):
        t = seg["ref"][r]
        t = t[(t >= WF_PRE) & (t + WF_POST < seg["x"].shape[1])][:n_spk]
        sig = seg["sigma_bp"][r]
        bx, bxh, yh = bp(seg["x"][r]) / sig, bp(seg["x_hat"][r]) / sig, seg["y_hat"][r] / sig
        cut = lambda s: np.stack([s[c - WF_PRE:c + WF_POST] for c in t])
        X_, XH_, YH_ = cut(bx), cut(bxh), cut(yh)
        for ax, model, colour in ((axes[i, 0], XH_, XH), (axes[i, 1], YH_, Y)):
            ax.plot(ms, X_.T, color=X, lw=0.5, alpha=0.35)
            ax.plot(ms, model.T, color=colour, lw=0.5, alpha=0.35)
            ax.plot(ms, X_.mean(0), color=X, lw=2, label="recording")
            label = "reconstruction x̂" if colour == XH else "spike decoder ŷ"
            ax.plot(ms, model.mean(0), color=colour, lw=2, ls="--", label=label)
            ax.axhline(-k, color="0.5", ls=":", lw=1)
            ax.axvline(0, color="0.85", lw=0.8, zorder=0)
        r_x = np.corrcoef(X_.ravel(), XH_.ravel())[0, 1]
        r_y = np.corrcoef(X_.ravel(), YH_.ravel())[0, 1]
        axes[i, 0].set_ylabel(f"electrode {r}\n{len(seg['ref'][r])} spikes\n(× noise σ)", fontsize=8)
        axes[i, 0].text(0.98, 0.05, f"r = {r_x:.2f}", transform=axes[i, 0].transAxes, ha="right", fontsize=8)
        axes[i, 1].text(0.98, 0.05, f"r = {r_y:.2f}", transform=axes[i, 1].transAxes, ha="right", fontsize=8)
    axes[0, 0].set_title("recording vs reconstruction (spike band 500–7500 Hz)", fontsize=9)
    axes[0, 1].set_title("recording vs spike decoder ŷ", fontsize=9)
    axes[0, 0].legend(fontsize=7, loc="upper right"), axes[0, 1].legend(fontsize=7, loc="upper right")
    axes[0, 0].text(WF_POST / HZ * 1e3, -k, f" −{k:g}σ = detection threshold", fontsize=7, color="0.4",
                    va="bottom", ha="right")
    for ax in axes[-1]:
        ax.set_xlabel("time from the NIX spike (ms)")
    fig.suptitle(f"Spike shapes on the {len(rows)} busiest electrodes: thin = single Good spikes, thick = mean.\n"
                 "The closer the dashed colour line follows the black one, the better the shape is kept; "
                 "a spike below the dotted line is detected.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.15 / fig.get_figheight()))   # room for the 2-line title
    return fig


def detection(seg, seconds=1.0):
    """Raster over 1 s: NIX Good times and the crossings on bp(x), bp(x̂) and ŷ (same sigma)."""
    rows = sorted(seg["ref"], key=lambda r: -len(seg["ref"][r]))[:8]
    n, k = int(seconds * HZ), env("K_THRESHOLD", 5.0)
    arms = {"NIX Good": None, "bp(x)": bp(seg["x"][rows, :n]), "bp(x̂)": bp(seg["x_hat"][rows, :n]),
            "ŷ": seg["y_hat"][rows, :n]}
    colours = {"NIX Good": "0.1", "bp(x)": "0.55", "bp(x̂)": XH, "ŷ": Y}
    fig, ax = plt.subplots(figsize=(14, 0.55 * len(rows) * 4 + 1))
    for i, r in enumerate(rows):
        for j, (arm, s) in enumerate(arms.items()):
            t = seg["ref"][r][seg["ref"][r] < n] if s is None else crossings(s[i], -k * seg["sigma_bp"][r])
            ax.eventplot(t / HZ * 1e3, lineoffsets=i * 4 + j, linelengths=0.8, colors=colours[arm])
    ax.set_yticks([i * 4 + 1.5 for i in range(len(rows))], [f"el {r}" for r in rows])
    ax.set_xlabel("time (ms)")
    ax.legend(handles=[plt.Line2D([], [], color=c, lw=2, label=a) for a, c in colours.items()], loc="upper right")
    ax.set_title(f"detection raster, -{k:g}σ (σ from bp(x))")
    return fig


def lfp_bands(seg):
    """Band envelopes of x and x̂ on one electrode over the whole segment, and the r per band."""
    c = seg["ch"]
    x1k, h1k = down_1k(seg["x"][c]), down_1k(seg["x_hat"][c])
    t = np.arange(len(x1k)) / 1000
    fig, axes = plt.subplots(len(LFP_BANDS) + 1, 1, figsize=(13, 1.6 * (len(LFP_BANDS) + 1)))
    for ax, (b, f) in zip(axes, LFP_BANDS.items()):
        ax.plot(t, lfp_band(x1k, *f), color=X, lw=0.8), ax.plot(t, lfp_band(h1k, *f), color=XH, lw=0.8)
        ax.set_ylabel(f"{b}\n{f[0]}-{f[1]} Hz", fontsize=8)
    r = [np.median(seg["lfp_r"][b]) for b in LFP_BANDS]
    axes[-1].bar(list(LFP_BANDS) + ["MUA"], r + [np.median(seg["mua_r"])], color=XH)
    axes[-1].set_ylim(min(0, *r), 1)
    axes[-1].set_ylabel("median r")
    fig.suptitle(f"LFP band envelopes, electrode {c} (black = x, red = x̂)")
    fig.tight_layout()
    return fig


def mua_trace(seg, n_el=6):
    """MUA of x and x̂ on the busiest electrodes, one panel each, with their correlation."""
    rows = _busiest(seg, n_el) or [seg["ch"]]
    fig, axes = plt.subplots(len(rows), 1, figsize=(13, 1.5 * len(rows) + 1.2), sharex=True, squeeze=False)
    for ax, r in zip(axes[:, 0], rows):
        a, b = mua(seg["x"][r]), mua(seg["x_hat"][r])
        t = np.arange(len(a)) / 1000
        ax.plot(t, a, color=X, lw=0.8, label="recording x")
        ax.plot(t, b, color=XH, lw=0.8, label="reconstruction x̂")
        ax.set_ylabel(f"electrode {r}\n(a.u.)", fontsize=8)
        ax.text(0.995, 0.85, f"r = {np.corrcoef(a, b)[0, 1]:.3f}", transform=ax.transAxes, ha="right", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="0.8", boxstyle="round,pad=0.2"))
    axes[0, 0].legend(fontsize=7, loc="upper left")
    axes[-1, 0].set_xlabel("time (s)")
    fig.suptitle("MUA = background activity of many neurons near the electrode "
                 "(500–5000 Hz, rectified, smoothed below 200 Hz).\n"
                 "Black = recording, red = reconstruction; r = their correlation (1 = identical).", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.15 / fig.get_figheight()))   # room for the 2-line title
    return fig


def psd(seg):
    """Median-over-electrodes Welch PSD of x and x̂, log-log, 1 Hz - 7.5 kHz."""
    n = min(seg["x"].shape[1], 2 ** 16)
    f, p = sps.welch(seg["x"], HZ, nperseg=n, axis=-1)
    _, ph = sps.welch(seg["x_hat"], HZ, nperseg=n, axis=-1)
    keep = (f >= 1) & (f <= 7500)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(f[keep], np.median(p, 0)[keep], color=X, label="x")
    ax.loglog(f[keep], np.median(ph, 0)[keep], color=XH, label="x̂")
    ax.set(xlabel="frequency (Hz)", ylabel="PSD (normalised²/Hz)", title="power spectrum, median over electrodes")
    ax.legend()
    return fig


def artifacts(seg):
    """The electrode with the most flagged time: trace with the flagged spans shaded."""
    c = int(np.argmax(seg["art"].sum(1)))
    t = np.arange(seg["x"].shape[1]) / HZ
    fig, ax = plt.subplots(figsize=(13, 3))
    ax.plot(t, seg["x"][c], color=X, lw=0.5, label="x")
    ax.plot(t, seg["x_hat"][c], color=XH, lw=0.5, alpha=0.7, label="x̂")
    ax.fill_between(t, 0, 1, where=seg["art"][c] > 0, color="#eda100", alpha=0.3,
                    transform=ax.get_xaxis_transform(), label="flagged")
    err = np.abs(seg["x_hat"] - seg["x"])
    inside = f"{err[seg['art'] > 0].mean():.3f}" if seg["art"].any() else "-"
    ax.set(xlabel="time (s)", title=f"electrode {c}: {100 * seg['art'].mean():.2f} % of the segment flagged · "
                                     f"|x̂ - x| inside {inside}, outside {err[seg['art'] == 0].mean():.3f}")
    ax.legend(loc="upper right")
    return fig


def train_vs_val(history):
    """Every loss term, train and val on the same axes: a growing gap means overfitting."""
    keys = [k for k in history["val"][0] if k.startswith("loss_")] if history["val"] else []
    if not keys:
        return None
    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.2), squeeze=False)
    for ax, k in zip(axes[0], keys):
        for split, colour in (("train", X), ("val", XH)):
            pts = [(h["step"], h[k]) for h in history[split] if k in h]
            if pts:
                ax.plot(*zip(*pts), "o-", ms=2, color=colour, label=split)
        ax.set(title=k, xlabel="step", yscale="log")
    axes[0, 0].legend()
    fig.tight_layout()
    return fig


def model_input(batch, burn_in, n=3):
    """What the model really sees (T9): x_n after the burn-in split, the mask m, the target y = m·bp(x)."""
    n = min(n, len(batch["x"]))
    x = batch["x"][:n].float().numpy()
    m = batch["m"][:n].float().numpy()
    good = batch["good"][:n].float().numpy()
    fig, axes = plt.subplots(3, n, figsize=(5 * n, 7), squeeze=False)
    for j in range(len(x)):
        c = int(np.argmax(good[j].sum(-1)))
        xs = x[j, c, burn_in:]
        t = _ms(len(xs))
        axes[0, j].plot(_ms(x.shape[-1], -burn_in), x[j, c], color=X, lw=0.5)
        axes[0, j].axvline(0, color=XH, ls="--", lw=1)
        axes[1, j].plot(t, m[j, c], color="#1baf7a")
        axes[2, j].plot(t, m[j, c] * bp(xs), color=Y, lw=0.7)
        axes[0, j].set_title(f"window {j}, electrode {c}")
    for i, label in enumerate(("x_n (burn-in | loss span)", "m", "y = m·bp(x)")):
        axes[i, 0].set_ylabel(label)
    for ax in axes[-1]:
        ax.set_xlabel("time (ms, 0 = start of the loss span)")
    fig.tight_layout()
    return fig


def monitor_figures(seg, history):
    return {"traces": lambda: traces(seg), "waveforms": lambda: waveforms(seg),
            "detection": lambda: detection(seg), "lfp_bands": lambda: lfp_bands(seg),
            "mua": lambda: mua_trace(seg), "psd": lambda: psd(seg), "artifacts": lambda: artifacts(seg),
            "train_vs_val": lambda: train_vs_val(history)}
