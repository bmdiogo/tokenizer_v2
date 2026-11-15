**Supplementary Figure | What the 32 numbers of a token hold.** **Draft: 2 of the 20 recordings of this group.** 125 Hz model, seed 1, P4, unseen patient,
120 s per recording, 2,880,000 tokens. Dimensions are not comparable between seeds.
**a**, One t-SNE of the z-scored tokens, coloured six ways (grey: no spike, so no value), 20,000 points: 10,000 tokens with a NIX Good spike and as many without
(spikes are 5.9 % of tokens, so they are oversampled to be visible), artifacts left out. Distances between
groups of points on a t-SNE map cannot be read quantitatively. Neighbours sharing the spike label (15 nearest, chance 0.50):
0.91 on the map, 0.90 in the 32 dimensions, 0.70 on the first two PCs. Last panel: spike tokens of the one
electrode with the most units, in its own map, because unit IDs mean nothing across electrodes.
**b**, Each row is a feature of the token's own 8 ms frame: slow voltage is the 1 kHz signal, an LFP signal is the signed band-passed voltage (zero-phase, so it also sees ahead, which a causal token cannot), and MUA and LFP power are log power. A single token holds the slow voltage; separating bands needs several tokens, as the decoder reads them. Left: |Spearman ρ| of each dimension alone. Right:
every dimension together, fitted on the first half of each recording and scored on the second: R², or AUC in italics for yes/no features
(0.50 = chance). Linear: ridge or logistic regression. Nonlinear: gradient-boosted trees. "—": not enough data of both classes.
**c**, PCA of the z-scored tokens. **d**, Mean token around a spike, on the first PCs of panel c; lag 0 is the token whose frame holds the spike.
Features computed with σ from bp(x); NIX Good times only.
