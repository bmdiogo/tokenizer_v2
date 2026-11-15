"""The codec: its size, causality, streaming and channel independence."""
import torch

from models.codec import Codec, CodecConfig


def test_default_is_the_125_hz_paper_model():
    m = Codec()
    assert m.ratio == 240                                 # 30 kHz / 240 = 125 Hz tokens
    # neural_tokenizer's weight-norm model had 12 422 372 = this + 6 400 Snake parameters (Snake removed)
    assert sum(p.numel() for p in m.parameters()) == 12_415_972


def test_62_hz_arm_appends_one_stage(monkeypatch):
    monkeypatch.setenv("STRIDES", "8,5,3,2,2")
    monkeypatch.setenv("WIDTHS", "32,64,128,256,512,512")
    monkeypatch.setenv("DILATIONS", "1,3,9|1,3,9|1,3|1|")
    assert Codec().ratio == 480


def small():
    return Codec(CodecConfig(strides="4,2", widths="4,8,8", dilations="1,3|1", latent_dim=4)).eval()


def test_future_frames_never_change_the_past():
    """Causal at the token level: an output sample may use the rest of its own frame (the one-frame
    algorithmic latency, 8 ms at 125 Hz), never a later frame."""
    m, x = small(), torch.randn(1, 2, 256)
    y = x.clone()
    y[..., 96:] = torch.randn_like(y[..., 96:])        # 96 = the start of frame 12 (8 samples per token)
    for a, b in zip(m(x), m(y)):
        n = 96 if a.shape[-1] == 256 else 12
        assert torch.equal(a[..., :n], b[..., :n])


def test_stream_equals_offline_and_burn_in_equals_the_slice():
    m, x = small(), torch.randn(2, 3, 512)
    full = m(x)
    state = {}
    chunks = [m.stream(x[..., t:t + 64], state) for t in range(0, 512, 64)]
    for i in range(3):
        assert torch.allclose(torch.cat([c[i] for c in chunks], -1), full[i], atol=1e-5)
    burnt = m(x, burn_in=256)
    assert torch.allclose(burnt[0], full[0][..., 256:], atol=1e-5)
    assert torch.allclose(burnt[1], full[1][..., 256:], atol=1e-5)
    xh, yh, z = m.reconstruct(x[0].numpy(), chunk=128)
    assert torch.allclose(torch.from_numpy(xh), full[0][0], atol=1e-5)


def test_each_electrode_is_its_own_sequence():
    m = small()
    x = torch.randn(1, 3, 128, requires_grad=True)
    m(x)[0][0, 1].sum().backward()
    assert x.grad[0, 0].abs().max() == 0 and x.grad[0, 2].abs().max() == 0 and x.grad[0, 1].abs().max() > 0


def test_burn_in_carries_no_graph_and_the_loss_span_trains():
    m = small().train()
    state = m.warm_up(torch.randn(1, 2, 128))
    assert all(not v.requires_grad for v in state.values())
    m(torch.randn(1, 2, 256), burn_in=128)[0].sum().backward()
    assert all(p.grad is not None for p in m.encoder.parameters())


def test_non_causal_ceiling_sees_only_its_reach_and_reconstructs_as_one_run():
    """CAUSAL=0 (baseline S1-N): the output sees the future, but no further than `future` samples and no further
    back than `past`, so the training windows (past + loss span + future) and the chunked `reconstruct`
    give exactly the output of one run over the whole recording (away from its two ends, where the chunks pad
    the input with zeros and one run pads every layer)."""
    m = Codec(CodecConfig(strides="4,2", widths="4,8,8", dilations="1,3|1", latent_dim=4, causal=False)).eval()
    x = torch.randn(1, 2, 1024)
    full = m(x)
    y = x.clone()
    y[..., 512:] = torch.randn_like(y[..., 512:])
    assert not torch.equal(m(y)[0][..., 511], full[0][..., 511])                         # it does look ahead
    assert torch.equal(m(y)[0][..., :512 - m.future], full[0][..., :512 - m.future])     # within its reach
    y = x.clone()
    y[..., :512] = torch.randn_like(y[..., :512])
    assert torch.equal(m(y)[0][..., 512 + m.past:], full[0][..., 512 + m.past:])
    n = 256                                                                             # one training window
    win = m(x[..., 400 - m.past - 8:400 + n + m.future], burn_in=m.past + 8)
    assert torch.allclose(win[0], full[0][..., 400:400 + n], atol=1e-5)
    assert torch.allclose(win[2], full[2][..., 50:50 + n // 8], atol=1e-5)
    xh, yh, z = m.reconstruct(x[0].numpy(), chunk=128)
    inner = slice(m.past, 1024 - m.future)
    assert torch.allclose(torch.from_numpy(yh)[:, inner], full[1][0][:, inner], atol=1e-5) and z.shape == full[2].shape


def test_default_non_causal_ceiling_fits_the_s8_windows():
    m = Codec(CodecConfig(causal=False))
    assert m.past <= 11520 and m.future % m.ratio == 0 and sum(p.numel() for p in m.parameters()) == 12_415_972
