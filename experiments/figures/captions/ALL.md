**Figure 1 | A causal codec turns 30 kHz recordings into 125 Hz tokens and returns the whole recording and a spikes-only trace.**
**a**, The encoder runs on the implant: each electrode's signal is compressed, causally and frame by frame, into
32 numbers every 8 ms. At the receiver, one decoder rebuilds the whole recording ($\hat x$) and a second decoder outputs
only the spikes of the NIX-curated (Good) units ($\hat y$). Illustration (not to scale): a participant with the Utah array (10 × 10, 400 µm pitch, 96 electrodes) in V1, near the occipital pole, cabled through its connector on the skull. Signal glyphs: the 100 ms of the recording in b around the spike in c ($x$ and $\hat x$ on one scale). **b**, One second of a recording
of patient P4, who was never seen in training: original $x$, reconstruction $\hat x$ and spike output $\hat y$ (each scaled to
its own range). **c**, One spike, 5 ms, in the spike band (500–7500 Hz), in units of σ (the robust noise SD of bp($x$)); the
dashed line is the −5σ detection threshold. **d**, How the training target of $\hat y$ is built: the NIX Good spike times
give a mask $m$ (2 ms Tukey windows), and the target is the recording's own spike band inside the mask. NIX waveforms are
never a target (they are whitened). Data b–d: 125 Hz model, seed 1, test group p4_unseen.


**Figure 2 | At 125 Hz the codec keeps 92–96 % of the spikes the original recording reveals; what it loses are spikes that only just cross the detection threshold.**
**a**, Retention (of the NIX Good spikes detected on the original, the share also detected on the output) per test group,
for the codec reconstruction $\hat x$, the spike decoder $\hat y$, PCA with the same number of values per 8 ms (32) and
plain sampling at the same rate. Codec: mean ± SD over 5 training seeds. **b**, Median retention per unit against the
unit's SNR (6–80σ bins with ≥ 5 units); dots: every P4 unit ($\hat x$). Units above ~12σ keep all their spikes.
**c**, Retention by how deep each spike's trough goes on the original (σ units; the detector threshold is 5σ); grey bars:
share of P4 spikes in each bin. The loss is concentrated in the first 0.5σ above the threshold, where background noise
decided whether the spike crossed at all; the codec removes that noise. Codec: mean of 5 seeds; PCA dashed.
**d**, Mean spike waveforms of eight P4 units from the clearest to the smallest (titles: SNR and the median per-spike
correlation of bp($\hat x$) with bp($x$)); solid: bp($x$) ± SD, dashed: bp($\hat x$), dotted: $\hat y$. **e**, Detection F1 (marker) with recall and
precision (ends of the line) on the original, the two codec outputs and PCA. Precision counts a detection as correct when
it matches any NIX unit (Good or Bad); recall is against Good units. Detector everywhere: −5σ crossings in 500–7500 Hz,
σ from the original, match ±0.5 ms. Test groups (recordings / Good units): P2 13 / 161, P3 18 / 562, P4 20 / 1 086
(P4: a patient never used in training).


**Figure 3 | Halving the token rate to 62.5 Hz costs about 3 points of retention and keeps unit sorting; at 31.3 Hz every feature degrades; linear compression at the same budgets loses most spikes.**
**a**, Retention against token rate for the codec ($\hat x$, $\hat y$; mean ± SD over 5 seeds per rate), PCA with the same
32 values per frame, and plain sampling at the same number of values per second. One marker shape per test group; groups
are never averaged. **b–g**, Detection F1 of $\hat y$, offline sorting accuracy of $\hat x$ (SpikeForest accuracy per NIX
Good unit, median), waveform correlation, MUA correlation and LFP band-envelope correlation (high γ, 80–200 Hz; δ,
1–4 Hz is r ≥ 0.999 everywhere and not shown) and the share of units that keep ≥ 95 % of their spikes. Grey band: the original recording, across the three groups (b, c). PCA dashed where it applies; sorting was not run
on the baselines. At 15.6 Hz the codec collapses (S6, validation, 15 k steps; not part of the test).


**Figure 4 | Besides spikes, the codec keeps the LFP and sorts units at least as well as the original; on a patient never seen in training the results are as good as on the training patients' later days.**
**a**, Correlation between the band envelopes (Hilbert amplitude) of the original and the codec output, per channel and
recording, for each LFP band and test group (box: quartiles, line: 10–90 %; a line cut by the axis has its end written); orange: PCA median. Linear compression keeps the slow LFP almost exactly, so
the LFP is a check that nothing is broken, not where the codec gains. **b**, Per-channel correlation of the multi-unit
activity (MUA: 500–5000 Hz rectified envelope, 1 kHz), codec vs PCA. The codec removes background activity above about
1.5 kHz, so MUA agreement depends on how much of a recording's MUA is that background. **c**, Two seconds of MUA on one
P4 electrode (r over the whole recording). **d**, Validation (earlier late days, used to select the model; grey open) against test (blue) for four
headline metrics, mean ± SD over 5 seeds; black: the original recording. **e**, Offline sorting accuracy per NIX Good
unit (SpikeForest accuracy after Hungarian matching), sorting the codec output against sorting the original with the same
sorter; every test group, 125 Hz seed 1.


**Figure 5 | The codec runs in real time, never uses future input, and settles within its receptive field after a cold start.**
**a**, What happens to each 8 ms frame at 125 Hz: its 240 samples arrive, the encoder and both decoders run on all 96
electrodes in one call (dark: median 5.6 ms, light: up to the 99th percentile 6.6 ms, H100), and the outputs for
that frame are released. The oldest sample of a frame is released 14.6 ms after it was recorded, the newest
6.6 ms after. **b**, Compute time per frame (cumulative share of 2 000 frames on H100, 500 on CPU) against each rate's
budget (the frame length). 99th percentiles — H100: 125 Hz 6.6 of 8 ms; 62.5 Hz 7.0 of 16 ms; 31.3 Hz 7.4 of 32 ms; CPU (20 threads): 125 Hz 38.6 of 8 ms; 62.5 Hz 55.6 of 16 ms; 31.3 Hz 88.2 of 32 ms.
**c**, Causality on real data (one P4 electrode): the input is replaced from 0 ms on; the outputs for the original and
the changed input are identical up to the start of the frame that contains the change (shaded from there on) (maximum difference
0e+00) — the first output sample that moves lies 6.7 ms
before the change, inside the same frame. Streaming frame by frame equals one offline pass (maximum difference
4e-03). **d**, After a cold start (empty state), the difference to a stream that was
already running (RMS, in units of the signal SD) falls to floating-point level once the receptive field is filled. Spikes kept in the first second, cold
vs running: 125 Hz 0.949 vs 0.949; 62.5 Hz 0.913 vs 0.914; 31.3 Hz 0.825 vs 0.826.


**Supplementary Figure 1 | The codec keeps the spectrum up to the spike band and drops the background above it.**
Welch power spectral density (8 192-sample windows, median over channels and recordings, first 30 s) of the original,
the codec at each token rate and PCA at the 125 Hz budget, per test group. Shaded: the spike band (500–7500 Hz). The
codec keeps the LFP and the spike band and removes background power above about 1.5 kHz (125 Hz) and 0.8 kHz (62.5 Hz),
which explains the MUA results of Fig. 4b and the detection gain of Fig. 2e.


**Supplementary Figure 2 | Spike sorting on the codec output finds the same units as on the original.**
**a, b**, Every −5σ crossing of the spike band on one P4 electrode (recording P4-test-44, row 93) as a
61-sample snippet projected on its first two principal components, coloured by the NIX Good unit it matches (±0.5 ms;
light grey: no Good unit), for the original recording (**a**) and the codec output $\hat x$ (**b**). **c**, Offline
sorting accuracy per NIX Good unit (SpikeForest accuracy after Hungarian matching; the same sorter on both signals),
codec output against original, for every unit of the three test groups (marker shape: group); dashed: equal accuracy.
125 Hz model, seed 1.


**Supplementary Figure 3 | Detection on one electrode.**
One second of one P4 electrode: NIX Good spike times and the −5σ crossings on the original, the codec reconstruction and
the spike decoder output. Hit: within ±0.5 ms of a Good spike; NIX Bad: matches only a spike of a Bad unit (not counted as
false); false alarm: matches no NIX spike; miss: a Good spike with no crossing.


**Supplementary Figure 4 | Waveforms of twelve P4 units and waveform correlation for every unit.**
**a**, Mean spike-band waveforms (σ units) on the original (black, ± SD), the codec reconstruction (dashed) and the spike
decoder (dotted) for twelve P4 units spanning the SNR range. **b**, Median per-spike correlation of bp($\hat x$) with bp($x$)
for every NIX Good unit of the three test groups against the unit's SNR (125 Hz, seed 1). NIX templates are not shown:
they are whitened and on another scale.


**Supplementary Figure 5 | Unit IDs can be given online, at the receiver, from the spike decoder output alone.**
**a**, 150 ms of one P4 electrode (recording P4-test-44, row 93; chosen by rule: of the
4 example electrodes, 1 where the online sorter holds templates for at least two NIX units, and there
the window with the most NIX Good spikes of at least two units). Top: the spike band of the original with the NIX Good spikes coloured
by unit (the offline, curated reference). Bottom: the spike decoder output $\hat y$ with every −5σ crossing coloured by
the online sorter's ID (nearest template; templates from the first 60 s), each ID drawn in the colour of the NIX unit
its spikes match most often; grey: no template close enough (unsorted). **b**, Online accuracy per NIX Good unit on
$\hat y$ (SpikeForest accuracy; box: quartiles, marker: median) for the three test groups; black line: median offline
accuracy of sorting the original recording. 125 Hz model, seed 1.
