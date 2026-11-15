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
