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
