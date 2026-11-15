"""The four loss terms."""
import torch

from utils.dsp import bp
from utils.losses import LossConfig, losses, stft_loss


def batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 3, 7680, generator=g)
    m = torch.zeros_like(x)
    m[..., 1000:1061] = 1
    m[..., 5000:5061] = 1
    return x, m


def test_a_perfect_output_costs_nothing():
    x, m = batch()
    out = losses(x.clone(), m * bp(x), x, m, LossConfig())
    assert all(float(v) < 1e-4 for v in out.values())


def test_spike_loss_without_background_is_the_wave_loss_on_y_hat():
    x, m = batch()
    x_hat = x + 0.3 * torch.randn_like(x)
    out = losses(x_hat, bp(x_hat), x, m, LossConfig(lambda_bg=0.0))
    assert torch.allclose(out["loss_spike"], out["loss_wave"], rtol=1e-4)


def test_background_weight_punishes_false_spikes():
    x, m = batch()
    noisy_y = m * bp(x) + 0.5 * (1 - m) * torch.randn_like(x)
    assert losses(x, noisy_y, x, m, LossConfig(lambda_bg=0.0))["loss_spike"] < 1e-5
    assert losses(x, noisy_y, x, m, LossConfig(lambda_bg=1.0))["loss_spike"] > 0.1


def test_a_silent_output_is_not_the_stft_optimum():
    x, _ = batch()
    w = (64, 128, 512, 2048)
    assert stft_loss(x, x, w) < 1e-6 < stft_loss(0.1 * x, x, w) < stft_loss(torch.zeros_like(x), x, w)
