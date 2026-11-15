"""The codec: one electrode at a time, 30 kHz -> 32 numbers every 8 ms -> two decoders.

    x [B, 96, T] --encoder--> z [B·96, 32, T/240] --reconstruction decoder--> x̂  (the whole recording)
                                                  └-spike decoder----------> ŷ  (Good-unit spikes only)

Every conv is weight-normed and padded on the LEFT only, so no output depends on a later frame:
an output sample waits at most for the end of its own frame (one-frame latency, 8 ms at 125 Hz).
That makes the model streamable: `warm_up` + `stream` give exactly the offline output.

CAUSAL=0 is the non-causal ceiling: the same layers with the padding split between the two
sides, so every output also sees the future. It cannot stream; it answers what the causal constraint costs.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from math import prod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from utils.config import HZ, E, ints


# ================= CONFIG — every value can be changed in .env =================
@dataclass
class CodecConfig:
    latent_dim: int = E("LATENT_DIM", 32)                  # numbers per token, per electrode
    strides: str = E("STRIDES", "8,5,3,2")                 # product = samples per token: 240 -> 125 Hz
    widths: str = E("WIDTHS", "32,64,128,256,512")         # input projection, then one per stride
    decoder_mult: float = E("DECODER_MULT", 1.5)           # decoder widths = encoder widths x this
    dilations: str = E("DILATIONS", "1,3,9|1,3,9|1,3|1")   # residual units per stage, '|' between stages
    activation: str = E("ACTIVATION", "elu")               # elu | gelu | none (none = a linear codec)
    causal: bool = E("CAUSAL", True)                       # False: the non-causal ceiling (centred padding)
# ===============================================================================


ACTIVATIONS = {"elu": nn.ELU, "gelu": nn.GELU, "none": nn.Identity}
# The non-linearity after each strided conv and inside each residual unit. Without one the whole
# codec is a single linear filter. ELU is SoundStream / EnCodec's choice and has no parameters.


class Causal(nn.Module):
    """Base of every conv. While streaming, `stream` = (state, key) and the layer keeps the last
    `context` inputs in state[key]; outside a stream it pads with zeros instead."""
    stream = None
    causal = True                                      # set False on every layer by Codec(causal=False)

    def history(self, x):
        state, key = self.stream
        past = state.get(key, x.new_zeros(*x.shape[:-1], self.context))   # a stream starts from zeros
        xx = torch.cat([past, x], -1)
        state[key] = xx[..., xx.shape[-1] - self.context:]
        return xx


class Conv(Causal):
    """Causal conv (weight norm) + optional activation."""

    def __init__(self, c_in, c_out, k, stride=1, dilation=1, act="none"):
        super().__init__()
        self.conv = weight_norm(nn.Conv1d(c_in, c_out, k, stride, dilation=dilation))
        self.act = ACTIVATIONS[act]()
        self.stride, self.context = stride, (k - 1) * dilation + 1 - stride

    def pad(self):
        """(left, right) zero padding outside a stream: all left when causal, split in half otherwise."""
        return (self.context, 0) if self.causal else (self.context // 2, self.context - self.context // 2)

    def forward(self, x):
        x = self.history(x) if self.stream else F.pad(x, self.pad())
        return self.act(self.conv(x))


class TConv(Causal):
    """Causal transposed conv (length x stride) + activation; the extra samples at the RIGHT are dropped."""

    def __init__(self, c_in, c_out, stride, act):
        super().__init__()
        self.conv = weight_norm(nn.ConvTranspose1d(c_in, c_out, 2 * stride, stride))
        self.act = ACTIVATIONS[act]()
        self.stride, self.context = stride, 1        # k - stride = stride extra samples = 1 input frame

    def forward(self, x):
        n = x.shape[-1] * self.stride
        if self.stream:                               # the previous frame's kernel reaches into this chunk
            return self.act(self.conv(self.history(x))[..., self.stride:self.stride + n])
        off = 0 if self.causal else self.stride // 2   # non-causal: drop the extra samples on both sides
        return self.act(self.conv(x)[..., off:off + n])


class Residual(nn.Module):
    """h + 1x1(act(dilated k7 conv(h))): refines, same length."""

    def __init__(self, c, dilation, act):
        super().__init__()
        self.f = nn.Sequential(Conv(c, c, 7, dilation=dilation, act=act), Conv(c, c, 1))

    def forward(self, x):
        return x + self.f(x)


def encoder_block(c_in, c_out, stride, dilations, act):   # length / stride, channels up
    return nn.Sequential(Conv(c_in, c_out, 2 * stride, stride, act=act), *[Residual(c_out, d, act) for d in dilations])


def decoder_block(c_in, c_out, stride, dilations, act):   # length x stride, channels down
    return nn.Sequential(TConv(c_in, c_out, stride, act), *[Residual(c_out, d, act) for d in dilations])


class Codec(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg = cfg or CodecConfig()
        s, w = ints(cfg.strides), ints(cfg.widths)
        dil = [ints(d) for d in cfg.dilations.split("|")]
        if len(w) != len(s) + 1 or len(dil) != len(s):
            raise ValueError("WIDTHS needs one entry more than STRIDES, DILATIONS one group per stride")
        self.ratio = prod(s)
        self.encoder = nn.Sequential(
            Conv(1, w[0], 1),
            *[encoder_block(w[i], w[i + 1], s[i], dil[i], cfg.activation) for i in range(len(s))],
            Conv(w[-1], cfg.latent_dim, 1))
        dw = [round(c * cfg.decoder_mult) for c in w]

        def decoder():                                    # deepest stage first
            return nn.Sequential(
                Conv(cfg.latent_dim, dw[-1], 1),
                *[decoder_block(dw[i + 1], dw[i], s[i], dil[i], cfg.activation) for i in reversed(range(len(s)))],
                Conv(dw[0], 1, 1))

        self.decoder, self.spike_decoder = decoder(), decoder()
        for m in self.modules():
            if isinstance(m, Causal):
                m.causal = cfg.causal
        self.past, self.future = self.reach(0), self.reach(1)

    def reach(self, side):
        """Input samples an output sample can see on one side (0 past, 1 future), rounded up to whole tokens.
        An upper bound: the reaches of all layers added up (a residual branch adds to its input's).
        Causal: future = 0, besides the wait for the end of the own frame."""
        total = 0
        for seq, rate in ((self.encoder, 1), (self.decoder, self.ratio)):
            for m in seq.modules():
                if isinstance(m, Conv):
                    total += m.pad()[side] * rate
                    rate *= m.stride
                elif isinstance(m, TConv):
                    total += rate if side == 0 or not m.causal else 0      # one input frame
                    rate //= m.stride
        return 0 if side and self.cfg.causal else -(-total // self.ratio) * self.ratio

    def run(self, x):
        """[B, C, T] -> (x̂, ŷ, z). Every electrode is its own sequence."""
        B, C, T = x.shape
        if T % self.ratio:
            raise ValueError(f"length {T} is not a multiple of {self.ratio} samples per token")
        z = self.encoder(x.reshape(B * C, 1, T))
        return self.decoder(z).reshape(B, C, T), self.spike_decoder(z).reshape(B, C, T), z

    def forward(self, x, burn_in=0):
        """Outputs on x[..., burn_in:]. The burn-in runs WITHOUT gradient and only fills the
        causal buffers with real history (R2D2-style), so training sees what streaming sees.
        Non-causal: x ends with `self.future` samples after the outputs; the last `self.past` samples
        of the burn-in and that future are context, run with the rest (there is no stream)."""
        if not burn_in:                               # the whole input, zero-padded (both sides if non-causal)
            return self.run(x)
        if not self.cfg.causal:
            if burn_in < self.past:
                raise ValueError(f"BURN_IN {burn_in} < the non-causal past reach {self.past}")
            p, n, r = self.past, x.shape[-1] - burn_in - self.future, self.ratio
            x_hat, y_hat, z = self.run(x[..., burn_in - p:])
            return x_hat[..., p:p + n], y_hat[..., p:p + n], z[..., p // r:(p + n) // r]
        state = self.warm_up(x[..., :burn_in])
        return self.stream(x[..., burn_in:], state)

    # ------------------------------------------------------------ streaming
    @contextmanager
    def streaming(self, state):
        layers = [(n, m) for n, m in self.named_modules() if isinstance(m, Causal)]
        for name, m in layers:
            m.stream = (state, name)
        try:
            yield
        finally:
            for _, m in layers:
                m.stream = None

    def stream(self, x, state):
        """Next chunk (a multiple of the stride product) after everything `state` has seen.
        `state` is updated in place; start a stream with {}."""
        if not self.cfg.causal:
            raise RuntimeError("the non-causal ceiling cannot stream")
        with self.streaming(state):
            return self.run(x)

    @torch.no_grad()
    def warm_up(self, x):
        state = {}
        self.stream(x, state)
        return state

    @torch.no_grad()
    def reconstruct(self, x, chunk=7680):
        """A whole recording x [C, T] (numpy, normalised), streamed causally chunk by chunk
        (non-causal: each chunk with its past and future context; within `past` / `future` of the recording's
        two ends the output differs slightly from one run, which pads each layer instead of the input).
        -> numpy (x̂ [C, T], ŷ [C, T], z [C, latent, T/ratio]); T is cut to a multiple of the ratio."""
        was, dev = self.training, next(self.parameters()).device
        self.eval()
        x = torch.as_tensor(x[:, :x.shape[1] // self.ratio * self.ratio], dtype=torch.float32)[None]
        state, outs = {}, []
        if not self.cfg.causal:                       # chunks with their past and future context, zeros at the ends
            xp = F.pad(x, (self.past, self.future + chunk))
            for t in range(0, x.shape[-1], chunk):
                o = self(xp[..., t:t + self.past + chunk + self.future].to(dev), self.past)
                outs.append([v.float().cpu() for v in o])
        else:
            for t in range(0, x.shape[-1], chunk):
                outs.append([o.float().cpu() for o in self.stream(x[..., t:t + chunk].to(dev), state)])
        self.train(was)
        n = x.shape[-1]                               # the last non-causal chunk runs past the end
        xh, yh, z = (torch.cat(o, -1) for o in zip(*outs))
        return xh[0, :, :n].numpy(), yh[0, :, :n].numpy(), z[..., :n // self.ratio].numpy()

    def describe(self):
        n = sum(p.numel() for p in self.parameters())
        return f"Codec: {n:,} parameters, {self.ratio} samples per token ({HZ / self.ratio:g} Hz at {HZ / 1000:g} kHz)"
