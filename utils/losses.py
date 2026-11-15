"""The training objective: two decoders, two targets, four terms, all on the loss span L.

    x̂ ← ℒ_time + λ_STFT·ℒ_STFT + ℒ_wave      target x, the whole normalised recording
    ŷ ← ℒ_spike                                target y = m·bp(x): the spike band on Good spikes, 0 elsewhere

No term masks artifacts: x̂ learns them too. The artifact flag only clears m (in the dataset).
"""
from dataclasses import dataclass

import torch

from utils.config import E, ints
from utils.dsp import bp, pool_1k


# ================= CONFIG — every value can be changed in .env =================
@dataclass
class LossConfig:
    lambda_time: float = E("LAMBDA_TIME", 1.0)
    # λ_STFT = 0: any STFT weight keeps corr < 0.99 in the T9 overfit (0.1: 0.982, 0.05: 0.986);
    # 0.05 was tested again in S7 (C′) and was worse on every metric.
    lambda_stft: float = E("LAMBDA_STFT", 0.0)
    lambda_wave: float = E("LAMBDA_WAVE", 1.0)
    lambda_spk: float = E("LAMBDA_SPK", 1.0)
    lambda_bg: float = E("LAMBDA_BG", 0.1)               # weight of the silence outside m in ℒ_spike (S5)
    mask_alpha: float = E("MASK_ALPHA", 0.25)            # edges of m: 0 box · 0.25 Tukey · 1 Hann (S7-E)
    mask_source: str = E("MASK_SOURCE", "nix")           # m from: nix (Good times) · crossings on bp(x) (S7-D)
    stft_30k: str = E("STFT_WINDOWS_30K", "64,128,512,2048")   # short = timing, long = frequency
    stft_1k: str = E("STFT_WINDOWS_1K", "64,128,256")          # the same on x at 1 kHz (LFP)
# ===============================================================================


def stft_loss(a, b, windows, eps=1e-5):
    """Sum over window lengths of the mean |log|STFT(a)| - log|STFT(b)||. The log gives the small,
    fast spike band the same weight as the big, slow LFP."""
    a, b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    total = a.new_zeros(())
    for w in windows:
        kw = dict(n_fft=w, hop_length=w // 4, window=torch.hann_window(w, device=a.device), return_complex=True)
        total = total + ((torch.stft(a, **kw).abs() + eps).log() - (torch.stft(b, **kw).abs() + eps).log()).abs().mean()
    return total


def losses(x_hat, y_hat, x, m, cfg):
    """x̂, ŷ, x, m: [B, C, L], normalised units. -> {"loss_total", "loss_time", ...} (fp32)."""
    with torch.autocast(x.device.type, enabled=False):
        x_hat, y_hat, x, m = x_hat.float(), y_hat.float(), x.float(), m.float()
        bp_x = bp(x)
        l_time = (x_hat - x).abs().mean()                                         # every sample
        l_stft = (stft_loss(x_hat, x, ints(cfg.stft_30k))
                  + stft_loss(pool_1k(x_hat), pool_1k(x), ints(cfg.stft_1k)))
        l_wave = (m * (bp(x_hat) - bp_x).abs()).sum() / m.sum().clamp_min(1)      # spike band, inside m only
        y = m * bp_x                                                              # ŷ's target
        w = m + cfg.lambda_bg * (1 - m)                                           # spikes 1, silence λ_bg
        l_spike = (w * (y_hat - y).abs()).sum() / w.sum().clamp_min(1e-6)
        rec = cfg.lambda_time * l_time + cfg.lambda_stft * l_stft + cfg.lambda_wave * l_wave
        total = rec + cfg.lambda_spk * l_spike
    return {"loss_total": total, "loss_time": l_time, "loss_stft": l_stft,
            "loss_wave": l_wave, "loss_spike": l_spike}
