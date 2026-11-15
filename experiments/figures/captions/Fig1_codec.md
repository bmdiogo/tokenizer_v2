**Figure 1 | A causal codec turns 30 kHz recordings into 125 Hz tokens and returns the whole recording and a spikes-only trace.**
**a**, The encoder runs on the implant: each electrode's signal is compressed, causally and frame by frame, into
32 numbers every 8 ms. At the receiver, one decoder rebuilds the whole recording ($\hat x$) and a second decoder outputs
only the spikes of the NIX-curated (Good) units ($\hat y$). Illustration (not to scale): a participant with the Utah array (10 × 10, 400 µm pitch, 96 electrodes) in V1, near the occipital pole, cabled through its connector on the skull. Signal glyphs: the 100 ms of the recording in b around the spike in c ($x$ and $\hat x$ on one scale). **b**, One second of a recording
of patient P4, who was never seen in training: original $x$, reconstruction $\hat x$ and spike output $\hat y$ (each scaled to
its own range). **c**, One spike, 5 ms, in the spike band (500–7500 Hz), in units of σ (the robust noise SD of bp($x$)); the
dashed line is the −5σ detection threshold. **d**, How the training target of $\hat y$ is built: the NIX Good spike times
give a mask $m$ (2 ms Tukey windows), and the target is the recording's own spike band inside the mask. NIX waveforms are
never a target (they are whitened). Data b–d: 125 Hz model, seed 1, test group p4_unseen.
