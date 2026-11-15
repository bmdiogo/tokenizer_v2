# Results ledger (tokenizer_v2)

> The lab notebook of the project, kept in chronological order while the experiments ran and
> published as it was. Commit hashes and job numbers refer to the private development history.
> Recording names are pseudonymised (`<patient>-<split>-<nn>`); `info/splits.csv` is not public.

Every number here comes from a JSON in this folder, and every JSON carries the split hash and the
git hash. These numbers are **not comparable** to neural_tokenizer's T1–T7: the input is raw
(not 2 Hz – 6 kHz), the split has no `p3_legacy`, the activation is ELU (not Snake), precision is
counted against NIX Good + Bad (S9), and the code is new.

Split: `info/splits.csv`, hash `8159c355670a` (175 cached recordings; val = p2_late 11,
p3_late 11; test = p2_final 13, p3_final 18, p4_unseen 20). Rebuilt on MN5 from the NS5 headers
(`splits_from_headers.csv`, hash `14ea01191d7d`): identical split / group / cached / n_channels /
buildable for all 494 recordings. Only `size_gb` differs (stat vs share table).

## Cache (30 Sep, raw input)

175 / 175 recordings, 123 GB. NIX: 5 726 Good units (5.87 M spikes) and 8 663 Bad units (3.05 M
spikes) on resolved electrodes. Artifacts (`cache_report.csv`): the hyper-synchronous rule does
almost all the flagging. Median ≈ 0.04 % of time; worst recordings: P3-train-101 (train)
40 %, P4-test-63 11.8 %, P3-test-27 11.1 %. Saturation and
common-mode are ≈ 0 everywhere. The design document warns that a loose hse rule removes real synchrony:
inspect the 40 % recording before trusting the rule.

## Pre-flight re-run (T1–T9)

| test | question | result | JSON |
|---|---|---|---|
| T1 | NIX waveform space and time offset | 100 Good units (3 P3 train recordings): r = 1.000 with bp(x), lag 0 (NIX time = sample 21), NIX / bp amplitude 0.32 (10–90 %: 0.23–0.38), 4.5 % of Good spikes overlap another on the same electrode | `t1.json` |
| T2 | receptive field, burn-in, memory per arm | RF 11 352 / 12 312 / 14 712 / 19 512 samples → B = 11 520 / 12 480 / 15 360 / 21 120 (unchanged). Params 12.42 / 18.19 / 23.96 / 29.73 M. Peak memory at batch 4: 10.2 / 10.7 / 13.2 / 18.2 GiB | `t2.json` |
| T3 | reference detector K (precision vs Good + Bad) | K = 5 stays: F1 0.761 (P 0.784, R 0.739; P vs Good only 0.668). K = 4.5: 0.758 | `t3.json` |
| T4 | causal, streamable, time per frame | Causality tests pass (future perturbation, prefix, streaming = offline). Streaming p50 / p99: **5.5 / 5.7 ms per 8 ms frame on an H100**, 32.3 / 35.4 ms on CPU (above budget). Params: encoder 4.50 M, each decoder 3.96 M | `t4.json` |
| T5 | linear floors (PCA, low-rate), 32 values per 8 ms equivalent at 4 kHz | F1 on p2_late / p3_late. Identity (ceiling) 0.731 / 0.767. **PCA 4 kHz 0.493 / 0.531**, 2 kHz 0.111 / 0.098, 1 kHz 0.018 / 0.018, 0.5 kHz 0.010 / 0.008. Low-rate 4 kHz 0.508 / 0.549, 2 kHz 0.108 / 0.089, ≤ 1 kHz 0.0. At 4 kHz, precision holds (0.92–0.96) and recall falls (0.34–0.38). PCA vs low-rate at 4 kHz: 0.015 / 0.018, below the old T7 floor (0.03) → no difference (the new floor is not measured yet) | `t5.json`, `eval/` |
| T6 | collapse per loss term / rate (+ S3 raw); no gradient clipping (`3116939`) | **S3: raw passes** (125 Hz, λ_STFT 0.1: corr 0.884). Corr at λ_STFT 0.05 / 0.1 / 0.2: 125 Hz 0.928 / 0.884 / 0.834; 62.5 Hz 0.866 / 0.802 / 0.743; 31.3 Hz 0.815 / 0.686 / **0.064** (collapse). Without clipping, λ 0.1 is lower at every rate (was 0.914 / 0.833 / 0.792 with clipping); λ 0.05 is best at every rate and the only weight at or above 0.85 at 62.5 Hz. STFT only collapses everywhere; time only never does | `t6.json` |
| T7 | noise floor over 5 seeds | not run: the gate failed on T9 (jobs 46834620, 46848664). The 5 runs and the scoring job were cancelled; resubmit when T9 passes | `t7.json` |
| T8 | online sorter settings (ideal y, 363 units) | default acc 0.678, 6.6 % unsorted. Calibration 30 s: 0.631 (worse by more than the floor) → 60 s stays. τ 99th pct: 0.702, 2.2 % unsorted (+0.024). α = 0: 0.677 (no drift term needed). min spikes 20: 0.686 | `t8.json` |
| T9 | recipe sanity checks; no gradient clipping, no EMA warm-up (`3116939`) | init OK (ℒ_time 0.749 vs mean abs x 0.744). Zero input OK (corr 0.001, F1 ŷ 0). **Overfit fails the design bar** (2 batches, 5 000 steps): corr **0.973** (bar 0.99), F1 x̂ **0.62** (ceiling 0.71), **F1 ŷ 0.04** (bar 0.9). ℒ_spike falls to 0.003 while F1 ŷ stays ≈ 0: the spike decoder is the blocker. The earlier run (1 000 steps, with clipping) gave corr 0.956, F1 x̂ 0.13, F1 ŷ 0.0 | `t9.json` |

### T9 diagnostic: what F1(ŷ) can reach on the two T9 batches (no training)

ŷ = a·y, with y = m·bp(x) the ideal target, scored with the T3 detector like ŷ:

| a | ℒ_spike | F1(ŷ) | P | R |
|---|---|---|---|---|
| 0 (silent) | 0.00291 | 0.000 | — | 0.000 |
| 0.1 | 0.00262 | 0.030 | 1.000 | 0.015 |
| 0.25 | 0.00218 | 0.068 | 0.941 | 0.035 |
| 0.5 | 0.00146 | 0.269 | 0.959 | 0.157 |
| 0.75 | 0.00073 | 0.584 | 0.974 | 0.417 |
| 1 (perfect) | 0 | **0.845** | 0.977 | 0.744 |

m covers 0.45 % of the samples, so a silent ŷ is already close to the minimum of ℒ_spike. A
perfect ŷ scores 0.845: the old bar (F1 > 0.9) was out of reach. New bar (design T9: "F1(ŷ) ≈ 1"
= ŷ reproduces y): **F1(ŷ) ≥ 0.9 · F1(y)**, and T9 trains until it passes (cap 30 k steps, with
y_scale = the least-squares scale of ŷ on y inside m logged every 1 000 steps). The 5 000-step run
ended at ℒ_spike 0.0026 ≈ 0.1·y, still falling.

### T9 at 30 k steps (bar 0.9 · F1(y), job 46848664): ŷ passes, corr does not

| step | corr(x̂, x) | F1 x̂ | F1 ŷ | y_scale |
|---|---|---|---|---|
| 1 000 | 0.957 | 0.14 | 0.00 | 0.00 |
| 5 000 | 0.974 | 0.61 | 0.04 | 0.66 |
| 10 000 | 0.977 | 0.70 | 0.57 | 0.89 |
| 15 000 | 0.978 | 0.75 | 0.75 | 0.95 |
| 30 000 | **0.982** | 0.77 | **0.77** | 0.95 |

ŷ was slow, not stuck: F1(ŷ) ≥ 0.76 (= 0.9 · 0.845) from step ~19 k. corr gains ~0.001 per 5 k
steps. **The corr bar is reachable**: on the same batches a linear code with the same budget gets
corr 0.991 (PCA 32/240 fitted on these frames 0.9913, on training sessions 0.9909; low-rate 4 kHz
0.9910). 97 % of the variance is below 300 Hz, so corr measures the LFP. The codec beats PCA on spikes
and loses to it on the slow signal: not a capacity limit. Untested suspect: the log-STFT term weights
every band equally (T6: time only has the highest corr). Next: one T9 with λ_STFT = 0.

### T9 with λ_STFT = 0 (diagnostic, job 46873289): passes at 11 k steps

Same T9, only `LAMBDA_STFT=0` for this run (`t9_stft0.json`; not the gate). corr 0.977 / 0.986 /
0.988 / 0.990 at 1 / 3 / 5 / 10 k steps; **at 11 k: corr 0.9902, F1 ŷ 0.762 (bar 0.760), F1 x̂ 0.78**;
y_scale 0.97. With the default λ_STFT 0.1, the same run reached corr 0.982 at 30 k. The log-STFT term
is what holds corr down; without it ŷ also learns sooner (F1 ŷ 0.76 at 10 k vs 19 k), and spike
detection on x̂ is no worse. ℒ_STFT itself stays at ≈ 6.2 (2.5 when trained): the spectrum on the log
scale is not fitted.

### T9 with λ_STFT = 0.05 (diagnostic, job 46887325): fails at 30 k steps

`LAMBDA_STFT=0.05` for this run only (`t9_stft005.json`). At 30 k: **corr 0.9864** (bar 0.99), **F1 ŷ
0.670** (bar 0.760), F1 x̂ 0.79, y_scale 0.92. corr gains ~0.0003 per 1 k steps at the end.

| λ_STFT | steps | corr | F1 x̂ | F1 ŷ | passes |
|---|---|---|---|---|---|
| 0.1 | 30 k | 0.982 | 0.77 | 0.77 | no |
| 0.05 | 30 k | 0.986 | 0.79 | 0.67 | no |
| 0 | 11 k | 0.990 | 0.78 | 0.76 | yes |

corr orders with λ_STFT as expected. F1 ŷ does not (one run each: the spread between runs is unknown).

**Decision: λ_STFT = 0 is the default** (`utils/losses.py`, `.env`): only it passes T9. ℒ_STFT stays in the
code; T7's `val_lfp/*` on unseen recordings shows what dropping it costs, and re-adding it is an S4
ablation. T6 now calls "time only" the default, and S3 is read from it.

## S5 · LR × scheduler search (submitted 2 Oct, jobs 46931646–53, scorer 46931654)

8 runs × 15 k steps at 125 Hz, seed 1 / data order 11 (= t7-s1, so the T7 floor applies), drawn with
`scripts/stage.py` (`s5_arms.json`, fixed before they start). Two changes from the design document, decided
with the user: **λ_STFT is fixed at 0** (T9; re-adding 0.05 is the S7-C arm) and **λ_bg {0.1, 0.3, 1.0}
is the extra random axis** (the ℒ_spike card's sweep, no extra runs). Runs wait for the gate; the scorer
(`scripts.compare --stage s5`) waits for the T7 scorer. Kept = within the T7 floor of the best on F1 ŷ and
waveform r in both validation groups.

| run | LR | weight decay | scheduler | λ_bg |
|---|---|---|---|---|
| s5-c1 | 9.74e-4 | 0 | wsd | 0.3 |
| s5-c2 | 1.43e-4 | 1e-3 | wsd | 0.1 |
| s5-c3 | 2.06e-4 | 1e-3 | constant | 0.1 |
| s5-c4 | 5.99e-5 | 0 | cosine | 1.0 |
| s5-c5 | 5.55e-4 | 1e-2 | cosine | 0.1 |
| s5-c6 | 1.82e-3 | 1e-4 | cosine | 1.0 |
| s5-c7 | 1.13e-3 | 0 | constant | 0.3 |
| s5-c8 | 6.07e-4 | 1e-4 | constant | 0.1 |

S4: the gate's T6 now also runs the 15.6 Hz arm (every S6 rate checked with weight norm).

## Gate, T7 and S5 runs (2 Oct; read on 6 Oct during MN5 maintenance)

**Gate passed** (job 46930807, git `f2ec85c`, `t9.json`, `t6.json`). T9 at λ_STFT 0 passes at 11 k steps: corr
0.990, F1 ŷ 0.776 (bar 0.760), F1 x̂ 0.791, y_scale 0.99. T6 corr after 150 steps, now with the 15.6 Hz arm (S4):

| λ_STFT | 125 Hz | 62.5 Hz | 31.3 Hz | 15.6 Hz |
|---|---|---|---|---|
| 0 (time only, default) | **0.952** | **0.898** | **0.852** | **0.815** |
| 0.05 | — | 0.872 | 0.762 | 0.710 |
| 0.1 | — | 0.858 | 0.737 | 0.372 ✗ |
| 0.2 | 0.853 | 0.743 | 0.098 ✗ | 0.063 ✗ |

The default escapes at every S6 rate (15.6 Hz just under the 0.85 harness bar, as 31.3 Hz was before). λ_STFT 0.1
collapses at 15.6 Hz.

**T7 and S5 trained, but neither scorer ran.** 13 runs × 15 k steps, ~75 min each (3.6 steps/s). Four
(t7-s3, s5-c1, s5-c2, s5-c4) reached step 15 000 and then crashed while W&B wrote an image: `FileNotFoundError` in a
`/tmp/…wandb-media/` folder shared by the jobs of one node. The crash came before `step_015000.pt` was written; `last.pt`
is at 14 k. The T7 scorer (`afterok` on all five runs) and the S5 scorer behind it can therefore never start. Fixed
in `8992dc0` (a private /tmp per job). The runs need resuming for their last 1 k steps when MN5 is back.

In-training validation (the 256 fixed windows, not `scripts.evaluate`; F1 of x itself 0.735):

| run | LR | wd | sched | λ_bg | F1 x̂ 15 k | F1 ŷ 5 / 10 / 15 k |
|---|---|---|---|---|---|---|
| t7-s1…s5 (defaults) | 1e-4 | 0 | constant | 0.3 | 0.716 – 0.731 | **0.000 / 0.000 / 0.000** (all five) |
| s5-c1 | 9.7e-4 | 0 | wsd | 0.3 | 0.736 | 0.54 / 0.69 / 0.74 |
| s5-c2 | 1.4e-4 | 1e-3 | wsd | 0.1 | 0.733 | 0.00 / 0.51 / 0.74 |
| s5-c3 | 2.1e-4 | 1e-3 | constant | 0.1 | 0.733 | 0.00 / 0.71 / **0.75** |
| s5-c4 | 6.0e-5 | 0 | cosine | 1.0 | 0.589 (14 k) | 0 / 0 / 0 |
| s5-c5 | 5.6e-4 | 1e-2 | cosine | 0.1 | 0.732 | 0.69 / 0.75 / **0.75** |
| s5-c6 | 1.8e-3 | 1e-4 | cosine | 1.0 | 0.734 | 0 / 0 / 0 |
| s5-c7 | 1.1e-3 | 0 | constant | 0.3 | 0.721 | 0.37 / 0.72 / 0.74 |
| s5-c8 | 6.1e-4 | 1e-4 | constant | 0.1 | 0.730 | 0.70 / 0.76 / 0.75 |

(c1, c2 15 k values are from the log line printed before the crash.) corr(x̂, x) = 0.99 and std ≈ 1.13 in every run.

- **ŷ stays silent at the defaults.** All five T7 seeds have F1 ŷ = 0 at every step, so T7 gives no noise floor for F1 ŷ.
  On full data, the silent ŷ ≈ 0 optimum of ℒ_spike (T9 diagnostic) holds for 15 k steps at λ_bg 0.3, LR 1e-4.
- **λ_bg decides whether ŷ learns.** λ_bg 1.0: silent at both LRs tried. λ_bg 0.3: learns only at LR ≈ 1e-3. λ_bg 0.1:
  learns in all four configs, to 0.74–0.75 by 15 k. A larger LR makes it learn sooner (c5, c8 at 5 k).
- **x̂ is already at the x ceiling.** F1 x̂ 0.72–0.74 vs 0.735 for x, except c4 (LR 6e-5, still rising).
- The four λ_bg 0.1 configs differ by ≤ 0.01 in F1 ŷ at 15 k, below the old T7 floor (0.03): **no difference** yet.
  The choice waits for `scripts.compare` (waveform r, sorting, MUA, both groups) and for a ŷ floor.

### Relaunch (6 Oct, after the maintenance; git `8a62fbd`)

- Resume to 15 k: t7-s3, s5-c1, s5-c2, s5-c4 (jobs 46997015–18). Old scorers 46930813 / 46931654 cancelled.
- **T7 re-run at a config where ŷ learns:** base **s5-c5** (LR 5.55e-4, wd 1e-2, cosine, λ_bg 0.1). Chosen
  among the four λ_bg 0.1 configs (no difference between them) because its ŷ learned earliest (0.69 at 5 k) and stayed
  flat at ~0.75, and cosine scales to the S8 length. s5-c5 is seed 1; t7b-s2…s5 are seeds 2–5 (jobs 46997019–22).
- Scorers: the default T7 → `t7_defaults.json` (x̂ floor at the defaults; ŷ is silent there), job 46997023; the c5 T7
  → `t7.json`, the floor `compare.py` reads, job 46997024; S5 → `s5.json`, job 46997025 (after the resumes and the c5 T7).
- Resumes done (6 Oct, 14–20 min each); all four now have `step_015000.pt`. Val at 15 k: t7-s3 F1 x̂ 0.722, ŷ 0.000;
  s5-c1 0.737 / 0.737; s5-c2 0.733 / 0.743; s5-c4 0.590 / 0.000 (same as the crashed runs' last log lines). The c5 T7 runs
  wait for GPUs (estimated start 8 Oct 12:00); time limits cut to 1 h 45 (runs) and 5 h (old T7 scorer) for backfill.

### T7 at the defaults scored; c5 T7 partly timed out (7 Oct)

`t7_defaults.json` (job 46997023, 4.5 h; t7-s1…s5, defaults; mean ± SD over 5 seeds at 15 k, seed spread at 15 k):

| metric | p2_late | p3_late |
|---|---|---|
| F1 x̂ (x: 0.731 / 0.767) | 0.728 ± 0.002 (0.006) | 0.779 ± 0.002 (0.004) |
| recall x̂ | 0.604 ± 0.003 (0.009) | 0.670 ± 0.003 (0.007) |
| F1 ŷ | 0.000 | 0.000 |
| waveform r | 0.935 ± 0.001 (0.003) | 0.945 ± 0.001 (0.002) |
| sorting acc | 0.496 ± 0.002 (0.005) | 0.583 ± 0.004 (0.010) |
| MUA r | 0.959 ± 0.001 (0.003) | 0.879 ± 0.002 (0.006) |

- x̂ detection is at the x ceiling (p3_late slightly above it). Every metric is still rising at 15 k (F1 x̂ 0.70 → 0.73 from
  10 k to 15 k on p2_late).
- **Floor definition.** `t7_noise_floor.py` takes the widest spread over all steps, which is the 5 k spread while the
  runs climb fast (F1 x̂ 0.052 at 5 k, 0.006 at 15 k). The design document judges S5–S7 at "the 15 k-step noise floor", so
  `compare.py` now uses the seed spread at the compared step (`git` after `d6d62e5`). The old T7 (~0.03) was also a
  5 k–15 k mix.
- c5 T7: t7b-s5 finished in 1 h 31; t7b-s2…s4 hit the 1 h 45 limit I set (10 k checkpoints saved). Resumed with 2 h (jobs
  47036930–32); T7 scorer 47036933 and S5 scorer 47036934 re-chained behind them.

## T7 at s5-c5 and S5 scored (8 Oct)

**T7 floor** (`t7.json`, job 47036933; s5-c5 + t7b-s2…s5, mean ± SD at 15 k, seed spread at 15 k = the floor S5–S7 use):

| metric | p2_late | p3_late |
|---|---|---|
| F1 ŷ (x: 0.731 / 0.767) | **0.722** ± 0.002 (0.005) | **0.766** ± 0.002 (0.006) |
| F1 x̂ | 0.738 ± 0.002 (0.004) | 0.776 ± 0.002 (0.004) |
| recall x̂ | 0.629 ± 0.005 (0.009) | 0.684 ± 0.004 (0.010) |
| waveform r | 0.939 ± 0.003 (0.007) | 0.950 ± 0.003 (0.006) |
| sorting acc | 0.514 ± 0.003 (0.007) | 0.587 ± 0.003 (0.008) |
| MUA r | 0.950 ± 0.004 (0.009) | 0.850 ± 0.010 (0.026) |

ŷ now has a floor. x̂ detection is at or above the x ceiling at 15 k. Against T7 at the defaults, c5 adds ŷ and costs
nothing on x̂ (F1 0.738 vs 0.728 on p2_late, above the floor; waveform r 0.939 vs 0.935, no difference).

**S5** (`s5.json`, job 47036934, 7 h; all 8 runs at 15 k, floor at 15 k). F1 ŷ / waveform r / F1 x̂ / sorting / MUA r:

| run | p2_late | p3_late |
|---|---|---|
| s5-c1 (9.7e-4, wsd, λ_bg 0.3) | 0.714 / 0.943 / 0.744 / 0.524 / 0.947 | 0.755 / 0.954 / 0.782 / 0.600 / 0.843 |
| s5-c2 (1.4e-4, wsd, 0.1) | 0.712 / 0.937 / 0.733 / 0.500 / 0.959 | 0.753 / 0.948 / 0.777 / 0.588 / 0.879 |
| s5-c3 (2.1e-4, const, 0.1) | 0.728 / 0.942 / 0.742 / 0.522 / 0.961 | 0.773 / 0.952 / 0.780 / 0.592 / 0.882 |
| s5-c4 (6.0e-5, cos, 1.0) | 0.000 / 0.905 / 0.594 / 0.350 / 0.857 | 0.000 / 0.917 / 0.634 / 0.397 / 0.856 |
| s5-c5 (5.6e-4, cos, 0.1) | 0.720 / 0.935 / 0.737 / 0.512 / 0.944 | 0.764 / 0.947 / 0.776 / 0.586 / 0.833 |
| s5-c6 (1.8e-3, cos, 1.0) | 0.000 / 0.933 / 0.738 / 0.521 / 0.912 | 0.000 / 0.946 / 0.779 / 0.595 / 0.756 |
| s5-c7 (1.1e-3, const, 0.3) | 0.718 / 0.939 / **0.751** / **0.541** / 0.930 | 0.756 / 0.951 / **0.788** / **0.616** / 0.809 |
| **s5-c8** (6.1e-4, const, 0.1) | **0.741** / **0.943** / 0.747 / 0.527 / 0.951 | **0.784** / **0.954** / 0.783 / 0.603 / 0.849 |

- **Kept (rule: within the floor of the best on F1 ŷ and waveform r in both groups): s5-c8 only.** Its F1 ŷ beats the
  next (c3) by 0.013 / 0.011, about 2 × the floor; on p3_late ŷ scores above the x ceiling (0.784 vs 0.767), as a
  supervised readout can.
- c7 (λ_bg 0.3, LR 1.1e-3) is best on x̂ detection and sorting, but its ŷ is 0.02 / 0.03 behind c8.
- LFP is not discriminative (r ≥ 0.991 in every band, every run).
- One seed per config: the floor comes from c5's seeds, so c8's margin assumes the same spread for c8.

**Decision: s5-c8 is the setting for every later run.** `.env`, `.env.example`, the MN5 `.env` and the code defaults
(`utils/train.py`, `utils/losses.py`) now hold LR 6.07e-4, weight decay 1e-4, constant schedule, λ_bg 0.1.
S6 and S7 compare against s5-c8 (same seed 1 / data order 11).

## S6 and S7 submitted (8 Oct, git `a2b4484`, all at the s5-c8 settings, seed 1 / data order 11, 15 k steps)

- **S6, token rate** (`s6_arms.json`, jobs 47073002–10; scorer 47073011, `--ref s5-c8` = the 125 Hz point): 62.5 / 31.3 /
  15.6 Hz, one ×2 stage appended per halving, burn-in from T2, each at LR × 0.5 / 1 / 2 (3.04e-4 / 6.07e-4 / 1.21e-3).
- **S7, one-factor ablations at 125 Hz** (`s7_arms.json`, jobs 47073013–18; scorer 47073019, `--ref s5-c8`):
  A `LAMBDA_SPK=0` · C `LAMBDA_STFT=0.05` · D `LAMBDA_WAVE=0` and `MASK_SOURCE=crossings` · E `MASK_ALPHA=0` and `=1`.
  S7-B (2 Hz – 6 kHz input) not run: it needs a second cache.
- Scorer limit raised to 16 h (S6 has ~60 evaluations; S5's 48 took 7 h).

## Chat A checks (8 Oct)

- **LFP after dropping ℒ_STFT: kept.** In S5 (all λ_STFT = 0), LFP band-envelope r ≥ 0.998 in every band (δ … high-γ),
  both groups, every run, including the silent-ŷ ones. LFP r is not discriminative here (T5: PCA also ≥ 0.998).
  MUA r is the sensitive one (0.95 / 0.85 for s5-c8); what STFT adds back is measured by S7-C.
- Submitted job 47073496 (git `5505b81`):
  - `art_check` on the 4 most-flagged **train / val** recordings (40 %, 5.9 %, 1.8 %, 1.8 % flagged; never a test
    recording, so the rule is not tuned on test). Question: do NIX Good spikes fall inside flagged spans more often than
    the flagged share of time (then the hse rule removes real bursts)? Figures `experiments/art_check_<name>.png`.
  - T8 re-scored on the **trained ŷ of s5-c8** (`t8_yhat.json`) instead of the ideal y. Training recordings, as in T8,
    so the ŷ is somewhat optimistic (the model was trained on them).

## Headline metric: retention (8 Oct, git `c0bc2b6`)

The hypothesis said "≥ 95 % of the NIX Good spikes". With the fixed detector, the original x itself finds only
~74 % of them (T3, recall 0.739), so that target is out of reach for any method, and the gap is detector/sorter
disagreement, not compression. New headline, **retention**: of the NIX Good spikes the detector finds on x
(±0.5 ms), the share it also finds on ŷ (and x̂). It is counted per spike (found on both), pooled with a bootstrap
CI, per unit (% of units ≥ 0.95) and by unit SNR. Spikes ŷ finds that x misses are reported apart (`extra_per_min`).
Absolute P / R / F1 vs NIX stay as reference rows. Code: `retention_counts`, `unit_retention` (`utils/metrics.py`),
`retention_<arm>` in `scripts/evaluate.py`, and in the T7 / compare metrics. Design document (Paper story, S0, S9)
updated.

Job 47083838: identity check (retention must be 1.0) and the T7 (c5) re-score, so retention gets its floor.
Old T7 evaluation files moved to `experiments/eval/old_t7c5/`. S5 was decided before this metric existed and is not
re-scored; S6 and S7 get it from their scorers.

## Overnight 8–9 Oct: retention floor, S7, Chat A checks

**Identity check passed:** the original scored against itself keeps 100 % (96 310 / 96 310 spikes, p2_late).

**T7 at c5 re-scored with retention** (`t7.json`, git `c0bc2b6`, mean of 5 seeds at 15 k; seed spread at 15 k):

| | p2_late | p3_late |
|---|---|---|
| retention ŷ | 0.794 (0.014) | 0.824 (0.014) |
| retention x̂ | 0.845 (0.013) | 0.872 (0.013) |
| F1 ŷ / F1 x̂ (unchanged) | 0.722 / 0.738 | 0.766 / 0.776 |

At 125 Hz and 15 k steps the codec keeps ~80–87 % of the spikes the original reveals: **below the 95 % target**.
F1 is at the ceiling while retention is not, so the codec loses some spikes and finds other NIX Good spikes the original
misses: exactly the swap that F1 hides and retention exposes. Full-length runs (S8) are where 95 % is tested.

**S7** (`s7.json`, reference s5-c8, single seed per arm; floors at 15 k from T7). Retention ŷ / F1 ŷ / F1 x̂ /
waveform r / MUA r, p2_late; p3_late in brackets where it changes the reading:

| arm | retention ŷ | F1 ŷ | F1 x̂ | waveform r | MUA r | verdict |
|---|---|---|---|---|---|---|
| reference s5-c8 | 0.829 | 0.741 | 0.747 | 0.943 | 0.951 | — |
| A · no ℒ_spike | 0 | 0 | 0.745 | 0.943 | 0.950 | x̂ **no difference** on every metric except sorting p2 (−0.011): training ŷ costs x̂ ~nothing → **keep the spike decoder** |
| C · λ_STFT 0.05 | 0.780 | 0.726 | 0.736 | **0.889** | **0.868** (0.641) | worse on every metric, MUA most → **λ_STFT = 0 confirmed** |
| D · no ℒ_wave | 0.695 | 0.679 | **0.643** | 0.903 | 0.938 | much worse (x̂ retention 0.656 vs 0.864) → **ℒ_wave is essential** |
| D · mask from crossings | **0.880** | 0.730 | 0.738 | 0.947 | 0.956 | retention better, F1 vs NIX worse (−0.011 / −0.018). See below |
| E · box mask (α 0) | 0.829 | 0.737 | 0.744 | 0.948 | 0.956 | **no difference** → keep α 0.25, "not sensitive" between 0 and 0.25 |
| E · Hann mask (α 1) | **0.565** | **0.576** | 0.752 | 0.929 | 0.934 | ŷ much worse → too soft an edge hides the spike's onset and tail |

The crossings arm needs care. Retention counts the spikes that the **same detector** finds on x, and this arm is trained
with exactly those crossings as its mask, so it is partly optimised for the metric by construction. Against the
independent labels (F1 vs NIX Good + Bad) it is worse than the NIX-labelled model. Reading: NIX labels help the
absolute spike score; crossings give a label-free model that is not far behind (useful as a no-labels baseline).

**Chat A checks** (job 47073496):
- **Artifact rule.** NIX Good spikes inside flagged spans vs the flagged share of time: 1.84 % vs 1.78 %, 1.83 vs
  1.82, 5.41 vs 5.88 on three recordings (enrichment 0.92–1.03: the rule does not target spikes). On
  P3-train-101 (40 % flagged) 94 % of the Good spikes fall inside flags (enrichment 2.3). Its figure shows a
  periodic interference on all 96 electrodes at once, so that recording is contaminated and the rule flags it
  correctly. Consequence: its spikes are cleared from m (no spike target there), which is the intended behaviour.
  **Rule unchanged.** (Train / val recordings only; no test recording was looked at.)
- **Online sorter on the trained ŷ of s5-c8** (`t8_yhat.json`, 363 units, training recordings): accuracy 0.536
  (ideal y: 0.678), unsorted 6.7 %. τ 99th percentile: 0.551 / 2.2 % unsorted (ideal: 0.702 / 2.2 %); α = 0: same as
  0.01. **Decision: τ = 99th percentile, α = 0** (`f732eb8`, deployed after the S6 scorer so S6 is scored with
  one sorter).

## S6 scored (9 Oct, `s6.json`, scorer 47073011)

Reference s5-c8 (125 Hz). All 9 runs finished; single seed per arm, scored on full validation days at 5 k / 10 k / 15 k,
floors at 15 k from T7. Retention ŷ / F1 ŷ / retention x̂ / waveform r / sorting / MUA r at 15 k, p2_late (p3_late):

| rate · LR | retention ŷ | F1 ŷ | retention x̂ | waveform r | sorting | MUA r |
|---|---|---|---|---|---|---|
| **125 Hz · s5-c8** | 0.829 (0.858) | 0.741 (0.784) | 0.864 (0.890) | 0.943 (0.954) | 0.527 (0.603) | 0.951 (0.849) |
| 62.5 Hz · ½× | 0.766 (0.796) | 0.714 (0.756) | 0.804 (0.840) | 0.911 (0.929) | 0.494 (0.577) | 0.863 (0.614) |
| **62.5 Hz · 1×** | 0.788 (0.821) | 0.727 (0.772) | 0.814 (0.853) | 0.911 (0.931) | 0.504 (0.587) | 0.783 (0.399) |
| 62.5 Hz · 2× | 0.750 (0.776) | 0.707 (0.746) | 0.780 (0.816) | 0.896 (0.919) | 0.484 (0.567) | 0.724 (0.194) |
| 31.3 Hz · ½× | 0.652 (0.676) | 0.637 (0.677) | 0.699 (0.742) | 0.852 (0.883) | 0.380 (0.452) | 0.638 (0.163) |
| **31.3 Hz · 1×** | 0.647 (0.669) | 0.641 (0.678) | 0.681 (0.720) | 0.846 (0.879) | 0.410 (0.490) | 0.647 (0.158) |
| 31.3 Hz · 2× | 0.510 (0.528) | 0.531 (0.564) | 0.558 (0.603) | 0.776 (0.819) | 0.293 (0.380) | 0.564 (0.123) |
| 15.6 Hz · ½× | 0.036 (0.027) | 0.049 (0.040) | 0.276 (0.312) | 0.632 (0.676) | 0.069 (0.094) | 0.420 (0.077) |
| 15.6 Hz · 1× / 2× | 0 | 0 | 0 | 0.035 / 0.014 | 0 | 0.109 / 0.062 |

LFP band r ≥ 0.998 in every arm, including the collapsed ones (the LFP survives every rate).

**Reading.**
- **At 15 k steps every lower rate is worse than 125 Hz, beyond the floor, on every spike metric.** Halving the rate
  costs retention ŷ −0.041 / −0.037 (floor 0.014), F1 ŷ −0.014 / −0.012 (floor 0.005), waveform r −0.03 / −0.02. The
  training-time numbers (8 Oct: "62.5 ≈ 125") were on short validation windows and hid this; the full-day scores do not.
- **MUA is the first feature to break**: 62.5 Hz p3_late 0.40 vs 0.85 (floor 0.026), while waveform r drops only 0.02.
  **Explained (9 Oct, training-time PSD and MUA plots of both runs at 15 k, `psd_125hz_*.png`, `psd_62.5hz_*.png`):**
  the codec does not reproduce the background "hash" (the many small, unsorted spikes of distant neurons) above a
  cutoff. x̂'s power follows x up to ~1.5 kHz at 125 Hz and only up to ~0.7–0.9 kHz at 62.5 Hz, then falls 1–3 orders
  of magnitude below x. The MUA band is 500–5000 Hz, so at 62.5 Hz most of it is gone: the MUA of x̂ keeps the peaks of
  the large sorted spikes (ℒ_wave protects them) but loses the background level between them. Where that background
  dominates the envelope (more electrodes on p3_late), MUA r collapses while waveform r barely moves. Cause: no loss
  asks for the hash. ℒ_wave covers only the mask (~0.5 % of samples) and ℒ_time is dominated by the large LFP power.
  This is the guess "adds high-frequency power" turned upside down: the decoder *removes* it. Consequence for the
  paper: the codec is a spike-preserving denoiser, not a faithful wideband codec; MUA in the unsorted sense is the
  first thing lost. Not a bug, so S8 proceeds unchanged. A loss that keeps the hash (e.g. a band-limited envelope or
  spectral term at 0.5–5 kHz) would be a new design decision, not part of the current design.
- **31.3 Hz**: retention −0.18 / −0.19. **15.6 Hz collapses** at LR 1× and 2× (x̂ ≈ silent, waveform r 0.03); at ½× it
  escapes only between 10 k and 15 k steps (waveform r 0.01 → 0.63). The one-batch S4 harness (corr 0.82, "no collapse")
  did not predict this: it overfits one batch, the full run has to generalise.
- **Learning rate**: 1× is best or tied at 62.5 and 31.3 Hz; 2× is worse everywhere; ½× is the only one that escapes at
  15.6 Hz. The winner's LR carries over except at the lowest rate.
- **Caveat: 15 k steps favour the fast learner.** Between 10 k and 15 k the 62.5 Hz run still gained +0.018 / +0.020 F1 ŷ
  and the 31.3 Hz run +0.055 / +0.054, against +0.007 / +0.005 for 125 Hz. Lower rates (more parameters, a longer
  burn-in) learn more slowly, so the gaps above are an upper bound on the full-length gap. The knee proper is measured
  in S8 at 200 k.
- **Parameter confound**: 62.5 Hz has 18.2 M parameters vs 12.4 M and is still worse, so the extra stage does not buy
  the result.

**Knee at 15 k:** no rate below 125 Hz stays within the floor; the first rate that breaks is 62.5 Hz (MUA first, then
retention). The first rate that fails outright is 15.6 Hz.

**S8 (approved 9 Oct):** 125 Hz s5-c8, 62.5 Hz 1×, 31.3 Hz 1×, 5 seeds each,
200 k steps → 15 runs. 15.6 Hz is reported as the failure point from S6, not re-run (the ½× run would be a single
attempt to rescue it; worth one run only if the budget allows).

**Deployed:** the τ 99th / α 0 sorter (`f732eb8`) is pushed to MN5 now that S6 is scored.

## S8 submitted (10 Oct, git `70b030c`)

`s8_arms.json`: ref (125 Hz, s5-c8 settings), s6-625hz-lr1, s6-313hz-lr1 × seeds 1–5 (data order 11–15), 200 k steps,
3 chained 24 h legs per run (jobs 47134897–47134941). Scorer = S9 (`compare --stage s9 --ref s8-ref`, steps 50 k /
100 k / 150 k / 200 k, job 47134942, 40 h limit). Floor beyond 15 k = spread of the five ref seeds (`fe4e05d`).

## S8 interim: p2_late complete (12 Oct, scorer 47134942 still on p3_late)

All 15 runs reached 200 k in their first leg (125 Hz ~17 h, 62.5 Hz ~18 h, 31.3 Hz ~19.5 h); the 30 empty follow-on
legs were cancelled to release the scorer. Mean of 5 seeds, seed spread (max − min) in brackets, p2_late, F1 ceiling
(original x) 0.731:

| rate | step | retention ŷ | retention x̂ | F1 ŷ | F1 x̂ | MUA r |
|---|---|---|---|---|---|---|
| 125 Hz | 50 k | 0.867 (0.005) | 0.896 | 0.754 | 0.752 | 0.952 |
| 125 Hz | 200 k | **0.893 (0.006)** | **0.918** | 0.766 | 0.765 | 0.939 |
| 62.5 Hz | 50 k | 0.816 (0.013) | 0.844 | 0.737 | 0.743 | 0.786 |
| 62.5 Hz | 200 k | **0.857 (0.010)** | 0.880 | 0.756 | 0.760 | 0.822 |
| 31.3 Hz | 50 k | 0.726 (0.072) | 0.753 | 0.690 | 0.693 | 0.674 |
| 31.3 Hz | 200 k | **0.765 (0.022)** | 0.790 | 0.709 | 0.715 | 0.692 |

At 200 k: waveform r (median over units) 0.980 / 0.960 / 0.917, amplitude ratio 0.971 / 0.939 / 0.880; offline sorting
on x̂ 0.567 / 0.556 / 0.486 (on the original x: 0.535); online sorting on ŷ 0.547 / 0.531 / 0.467; LFP r ≥ 0.983 in every
band at every rate. Retention ŷ by unit SNR at 200 k (median over units):

| SNR (σ) | 6–9 (25 units) | 9–12 (105) | 12–18 (20) | ≥ 18 (27) |
|---|---|---|---|---|
| 125 Hz | 0.84 | 0.88 | **0.997** | **1.0** |
| 62.5 Hz | 0.79 | 0.84 | 0.996 | 1.0 |
| 31.3 Hz | 0.69 | 0.73 | 0.976 | 0.987 |

**Reading (p2_late only; p3_late pending).**
- **Long training helps every rate**: retention ŷ +0.064 at 125 Hz from 15 k (0.829) to 200 k, still rising at 200 k
  (+0.006 per 50 k).
- **The 95 % target holds for units ≥ 12σ at 125 and 62.5 Hz (≈ 1.0) and fails for the small units (6–12σ: 0.84–0.88)**,
  which are 130 of the 177 units, so the pooled number stays at 0.89. The loss is concentrated where the spike is close
  to the threshold. Likely cause (not yet checked spike by spike): a small amplitude loss (ratio 0.97) moves a share of them just below −5σ.
- **Rates.** 62.5 Hz vs 125 Hz at 200 k: retention −0.036 (floor = ref seed spread 0.006) → still worse, the gap barely
  closed (−0.041 at 15 k). 31.3 Hz: −0.128 (was −0.18). F1 ŷ at 62.5 Hz −0.010, above the ceiling of the original
  either way (0.756 vs 0.731).
- **Sorting on x̂ beats sorting on x** at 125 and 62.5 Hz (0.567 / 0.556 vs 0.535): the reconstruction removes the
  background hash (see MUA above), and the sorter works better on the cleaner trace.
- **MUA at 125 Hz falls slightly with training** (0.952 → 0.939): the codec keeps cleaning the background as it trains.

## S8 / S9 scored (12 Oct, `s9.json`, scorer 47134942, 18.6 h, no failed run)

Mean ± SD of 5 seeds at 200 k; floor = spread of the five 125 Hz seeds (p2 / p3: retention 0.006 / 0.006, F1 ŷ 0.004 /
0.004, waveform r 0.004 / 0.003). Verdict vs 125 Hz in the last column. F1 ceiling of the original: 0.731 / 0.767.

| | rate | retention ŷ | retention x̂ | F1 ŷ | F1 x̂ | waveform r | sorting x̂ | MUA r | vs 125 Hz |
|---|---|---|---|---|---|---|---|---|---|
| p2_late | 125 Hz | **0.893 ± 0.002** | **0.918** | 0.766 | 0.765 | 0.980 | 0.567 | 0.939 | — |
| | 62.5 Hz | 0.857 ± 0.004 | 0.880 | 0.756 | 0.760 | 0.960 | 0.556 | 0.822 | worse; sorting no difference |
| | 31.3 Hz | 0.765 ± 0.009 | 0.790 | 0.709 | 0.715 | 0.917 | 0.486 | 0.692 | worse on all |
| p3_late | 125 Hz | **0.924 ± 0.002** | **0.947** | 0.810 | 0.800 | 0.986 | 0.625 | 0.870 | — |
| | 62.5 Hz | 0.893 ± 0.005 | 0.915 | 0.799 | 0.803 | 0.971 | 0.621 | 0.584 | worse; sorting no difference, F1 x̂ +0.003 |
| | 31.3 Hz | 0.794 ± 0.011 | 0.827 | 0.749 | 0.762 | 0.937 | 0.567 | 0.164 | worse on all |

Retention ŷ by unit SNR, p3_late at 200 k (median over units; p2_late in the interim entry above, same pattern):

| SNR (σ) | 6–9 (40 units) | 9–12 (234) | 12–18 (65) | ≥ 18 (40) | % units ≥ 0.95 |
|---|---|---|---|---|---|
| 125 Hz | 0.89 | 0.92 | **1.0** | **1.0** | 34.8 |
| 62.5 Hz | 0.85 | 0.88 | 1.0 | 1.0 | 31.7 |
| 31.3 Hz | 0.74 | 0.78 | 0.98 | 0.98 | 24.6 |

Offline sorting on x (the original) for reference: 0.535 / 0.599; online on ŷ at 125 Hz: 0.547 / 0.615.

**Reading.**
- **125 Hz is close to the 95 % target and meets it for clear units.** Pooled retention x̂ 0.918 / 0.947, ŷ 0.893 / 0.924;
  units ≥ 12σ keep ~100 % (median per SNR bin) in both groups. The shortfall is in the 6–12σ units (most units).
- **The knee.** 62.5 Hz is worse beyond the floor on retention (−0.036 / −0.031), F1 ŷ, waveform r and MUA, and *no
  different* on sorting in both groups. 31.3 Hz is worse on everything. So at the full horizon: halving the rate keeps
  the units (sorting) and the clear spikes, loses ~3 points of small-spike retention and much of the MUA; quartering it
  loses on every feature. Both stay far above linear compression at the same budget (T5).
- **Selection (S9).** Every metric rises monotonically to 200 k in every arm and group (trend in `s9.json`), so the
  checkpoint per rate is **step 200 000 (EMA)**. Each rate has one configuration (s5-c8 settings, LR 1×). All 5 seeds
  go to the test and are reported as mean ± SD (S8: distributions, not single runs).

## S10 submitted (13 Oct, git `97b540a`) — test once

Pre-test check (no change made): the numbers support the paper (125 Hz retention x̂ 0.92 / 0.95, clear units ~1.0,
F1 and sorting above the original, a clean knee, ablations, seeds ≪ gaps). The pooled 95 % is not met and is reported
as such; longer training (+0.006 / 50 k) would not reach it, anything else is a design change. Protocol: recordings last
120–175 s, so EVAL_SECONDS = 120 is the whole recording, as on validation; streaming starts from an empty state (same
for every arm, stated in Methods).

Jobs: scorer 47235558 (`compare --stage s10`, the 15 frozen checkpoints of `s10_frozen.json` at 200 k, p2_final /
p3_final / p4_unseen, floor = spread of the five 125 Hz seeds per group); baselines 47235559 (identity, PCA and low-rate
at F 240 / 480 / 960, K 32, per test group, no sorting). **Nothing is re-tuned after these numbers.**

## Which spikes are lost? (13 Oct, validation only, job 47243955)

Question: could a model change (e.g. attention, cut in v1 on one run, c12) lift retention to 95 %, or are the lost
spikes the ones that only just crossed −5σ on the noisy original? For every NIX Good spike found on x (outside
artifacts): its trough depth on bp(x) in σ (min within ±0.5 ms), the output trough there, kept or not. One checkpoint,
s8-ref-s1 at 200 k (125 Hz), p2_late / p3_late. Script and data: `experiments/lost_spikes.py`, `lost_spikes.json`.
Sanity: pooled retention 0.915 / 0.944 (x̂) and 0.893 / 0.924 (ŷ), the S8 numbers.

Retention by trough depth on x (p2_late / p3_late):

| depth on x (σ) | share of found spikes | x̂ | ŷ |
|---|---|---|---|
| 5–5.25 | 8.0 % / 7.4 % | 0.48 / 0.58 | 0.43 / 0.51 |
| 5.25–5.5 | 7.8 % / 7.4 % | 0.76 / 0.84 | 0.70 / 0.78 |
| 5.5–6 | 14.8 % / 13.8 % | 0.90 / 0.94 | 0.86 / 0.91 |
| 6–7 | 22.2 % / 21.8 % | 0.97 / 0.98 | 0.95 / 0.97 |
| 7–9 | 22.3 % / 24.3 % | 0.99 / 1.00 | 0.98 / 0.99 |
| ≥ 9 | 24.9 % / 25.3 % | 1.00 / 1.00 | 0.99–1.00 / 1.00 |
| **≥ 5.5 (0.5σ margin)** | 84 % / 85 % | **0.971 / 0.985** | **0.955 / 0.973** |
| **≥ 6 (1σ margin)** | 69 % / 71 % | **0.985 / 0.993** | **0.976 / 0.986** |

Lost spikes: median depth on x 5.21–5.32σ; 71 % / 77 % (x̂) and 65 % / 70 % (ŷ) of them lie within 0.5σ of the
threshold, 84–91 % within 1σ. On the output their trough is at a median 4.7σ, i.e. they miss the threshold narrowly.
Spikes that are kept keep 99 % of their trough depth (median ratio 0.990–0.995), so there is no general shrinkage
(this replaces the unverified "amplitude ratio 0.97" explanation). ŷ only: the bottom 10–25 % of its lost spikes have
an output trough near 0, i.e. ŷ is silent there (≈ 1–2 % of found spikes): a small real modelling miss.

**Reading.**
- The gap to 95 % is the threshold margin, not a modelling gap. Noise on bp(x) has SD 1σ per sample, so a spike whose
  trough is 5.1σ crossed −5σ partly by chance; the codec removes the background hash (PSD result), and with it the
  part of the noise that pushed it over. Any denoiser scored against detections on the noisy original loses these.
- **≥ 0.95 holds for every spike with a 0.5σ margin, for both outputs in both groups** (x̂ 0.971 / 0.985, ŷ 0.955 /
  0.973; 84–85 % of the spikes the original reveals). That is the honest form of the 95 % claim.
- Attention would add long-range context, not this; it is not re-tested (it would also need its own S5–S8). Listed as
  future work only for ŷ's silent misses.
- One seed and validation only; an explanation of the headline, not a replacement for it. S10 is untouched.

## S10 scored (14 Oct) — the test, once. Nothing is re-tuned after this.

Scorer 47235558 (13 h 24 min) and baselines 47235559 (8 h 17 min) both COMPLETED; 15 / 15 checkpoints scored, none
failed (`s10.json`, `eval/s10_*`, `eval/{identity,pca,lowrate}_F*_K32_<group>.json`). Mean ± SD over 5 seeds; the
floor per group is the spread of the five 125 Hz seeds there. Test groups: p2_final 13 recordings / 161 Good units,
p3_final 18 / 562, p4_unseen 20 / 1 086 (P4: a patient never seen in training, 2024).

**125 Hz (p2_final / p3_final / p4_unseen):**

| | p2_final | p3_final | p4_unseen |
|---|---|---|---|
| retention x̂ | **0.921 ± 0.003** | **0.934 ± 0.003** | **0.957 ± 0.002** |
| retention ŷ | **0.894 ± 0.003** | **0.909 ± 0.003** | **0.936 ± 0.002** |
| units ≥ 12σ (median, x̂ / ŷ) | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |
| units 6–12σ (median, x̂ / ŷ) | 0.90–0.92 / 0.84–0.89 | 0.91–0.94 / 0.87–0.91 | 0.92–0.95 / 0.89–0.93 |
| % units ≥ 0.95 (x̂ / ŷ) | 45 / 40 | 42 / 29 | 60 / 46 |
| F1 x (original) | 0.746 | 0.716 | 0.776 |
| F1 x̂ / ŷ | 0.786 / 0.790 | 0.746 / 0.755 | 0.797 / 0.799 |
| false alarms / min x / x̂ / ŷ | 177 / 71 / 43 | 110 / 51 / 25 | 191 / 127 / 96 |
| sorting: x offline / x̂ offline / ŷ online | 0.552 / 0.604 / 0.603 | 0.564 / 0.585 / 0.571 | 0.567 / 0.595 / 0.573 |
| waveform r / amplitude ratio | 0.983 / 0.978 | 0.985 / 0.981 | 0.987 / 0.985 |
| LFP r (lowest band) / MUA r | 0.991 / 0.995 | 0.999 / 0.868 | 0.999 / 0.923 |

**The knee (vs 125 Hz, beyond the floor in every group unless stated):**
- 62.5 Hz: retention ŷ 0.860 / 0.877 / 0.905 (−0.03), x̂ 0.883 / 0.900 / 0.924; waveform r −0.014 to −0.018; **sorting no
  difference in all three groups**; F1 x̂ no difference on p3 and p4; MUA no difference on p2, lost on p3 (0.574) and p4
  (0.809).
- 31.3 Hz: worse on every metric in every group (retention ŷ 0.759 / 0.778 / 0.814; sorting 0.529 / 0.533 / 0.543).
- The probe (secondary) is higher at the lower rates, as on validation (0.03–0.12 at 125 Hz): it is not a spike metric.

**Linear compression at the same budgets (32 numbers per frame; x̂ retention):** PCA 0.531 / 0.472 / 0.525 at 125 Hz,
0.096 / 0.065 / 0.078 at 62.5 Hz, 0.012 / 0.011 / 0.011 at 31.3 Hz; low-rate sampling 0.550 / 0.495 / 0.553, ~0.08,
0.000. F1 PCA 0.566 / 0.489 / 0.567 at 125 Hz. Identity: retention 1.000 and F1 = the x ceiling in every group (check).

**Reading.**
- **The validation results transfer.** On the later days of P2 and P3 retention matches validation (x̂ 0.921 vs
  0.918; 0.934 vs 0.947; ŷ 0.894 vs 0.893; 0.909 vs 0.924). On the **unseen patient P4 it is the highest of all**
  (x̂ 0.957, ŷ 0.936): nothing in the codec is specific to the training patients.
- **Pooled 95 %:** met by x̂ on p4_unseen only; not by ŷ in any group. Clear units (≥ 12σ) keep ~100 % everywhere;
  the gap is in the small units, as the lost-spike diagnostic explains (spikes that only just cross −5σ on x).
- **F1 and sorting on the codec beat the original in every test group**; ŷ halves the false alarms or better.
- **The knee holds out of sample:** 62.5 Hz keeps the units (sorting no different) and loses ~3 points of
  small-spike retention and part of the MUA; 31.3 Hz loses everything; all far above linear compression (PCA at
  125 Hz keeps about half the spikes; at 62.5 Hz under 10 %).
- Seed SD ≤ 0.003 at 125 Hz. The smallest gap claimed (62.5 vs 125 Hz retention, −0.03) is 4–8× the floor of its group (≤ 0.008).

## S11 figure data (15 Oct, jobs 47288134–47288139, git `9919bea`, test groups, frozen S10 models)

All six COMPLETED (3–66 min). Arrays in `experiments/figures/data/`; figures by `python -m scripts.figures`.
Nothing here is re-tuned: these are the S10 models read on the S10 test groups.

- **Streaming (`streaming.json`, one P4 recording, 96 electrodes per call):** per frame, H100 125 Hz median 5.6 ms,
  p99 6.6 ms of an 8 ms budget; 62.5 Hz p99 7.0 of 16 ms; 31.3 Hz p99 7.4 of 32 ms → real time at every rate.
  CPU (20 threads) p99 38.6 / 55.6 / 88.2 ms → not real time at any rate. Streaming = offline pass (max
  difference 4e-3 at 125 Hz); a changed input moves nothing before the frame that holds it (max 0.0).
- **Timing after the app's optimisations (`timing.json`, job 47347166, 16 Oct; 125 Hz, 96 electrodes, one 8 ms frame
  per call, same P4 recording):** p50 / p99 ms per frame, H100 vs CPU (20 threads):

  | pipeline | H100 | CPU |
  |---|---|---|
  | as trained (= S11 above) | 5.58 / 6.54 | 36.2 / 39.3 |
  | weight norm folded into plain weights (same output, max diff 0.0) | 4.03 / **4.74** | 32.6 / 34.8 |
  | folded, encoder + spike decoder only (the app; x̂ decoded only where shown) | 2.82 / **3.38** | 20.2 / 21.2 |

  On the GPU the encoder + ŷ pipeline uses 42 % of the 8 ms budget at p99 (as trained: 82 %). On a server CPU it is
  still 2.6 × over budget at 96 electrodes, so CPU real time is not claimed.
- **Sorting per unit (`sorting.json`, seed 1):** units sorting as well or better on x̂ than on x: P2 83 %, P3 74 %,
  P4 81 %. Online IDs on ŷ: median accuracy 0.60 / 0.57 / 0.57.
- **MUA example:** one P4 electrode, r = 0.93 over the whole recording.

## Label-free retention on S10 (submitted 15 Oct, git `05165f8`, jobs 47349824 / 47349826 / 47349828)

Asked by the supervisor: how much of the raw signal is kept, x vs x̂, without curated spikes. `crossings_<arm>`: of
the −5σ crossings on bp(x) on **every** electrode (outside artifacts, sigma from bp(x)), the share each output also
crosses within ±0.5 ms; `new_per_min` = crossings on the output that x does not have. Hash and NIX Bad units count,
so ŷ (trained on Good times only) is low by design; the label-free number that means something is x̂'s. The 15 frozen
S10 checkpoints at 200 k, each test group, no sorting; written to `eval/crossings/`, the S10 JSONs are untouched.
A new metric on frozen models, nothing re-tuned (S10 holds).

**Results (done 16 Oct).** 5-seed mean ± SD, p2_final / p3_final / p4_unseen. "New" = the output's crossings that x
lacks, as % of x's crossings (the rates in the JSON are per recording-minute over all 96 electrodes; 2 min × 13 / 18 /
20 recordings). x has 444 655 / 520 033 / 1 872 842 crossings.

| arm | kept by x̂ | new on x̂ | kept by ŷ | new on ŷ |
|---|---|---|---|---|
| 125 Hz (s8-ref) | **0.738 / 0.786 / 0.862** (± 0.003) | 7 / 4 / 4 % | 0.640 / 0.704 / 0.811 | 4 / 3 / 3 % |
| 62.5 Hz | 0.647 / 0.715 / 0.799 (± 0.004) | 9 / 4 / 5 % | 0.599 / 0.673 / 0.766 | 4 / 3 / 4 % |
| 31.3 Hz | 0.550 / 0.630 / 0.695 (± 0.012) | 11 / 6 / 7 % | 0.507 / 0.589 / 0.664 | 4 / 3 / 4 % |

What x̂ keeps of x without labels, at 125 Hz: LFP r ≥ 0.992 in every band, MUA r 0.995 / 0.868 / 0.923, crossings
0.738 / 0.786 / 0.862, waveform r 0.983 / 0.985 / 0.987. On curated Good spikes (retention): 0.921 / 0.934 / 0.957.
The label-free figure is lower because all crossings include hash, Bad units and noise at the threshold, which the
codec drops (the same denoising that lifts F1 on x̂ above x). Each halving of the rate costs 7–10 points.

**PCA, same budget (jobs 47373185 / 86 / 87, 16 Oct; `eval/crossings/pca_F*_K32_*.json`, `--no-sorting`).** Label-free
crossings kept at 125 / 62.5 / 31.3 Hz budgets: **0.377 / 0.343 / 0.403**, 0.070 / 0.057 / 0.061, 0.010 / 0.009 /
0.009. The codec keeps about twice as many at 125 Hz (0.738 / 0.786 / 0.862) and 10× more at 62.5 Hz: the same
ratio as on curated spikes (PCA retention 0.53 / 0.47 / 0.52).

## ŷ operating point: threshold sweep (supervisor, 16 Oct; git `db66031` + fix, jobs 47371759 quick / 47371760 full)

`figure_data threshold`: crossings of −Kσ (σ from bp(x)) on ŷ (125 Hz, 5 seeds) and on bp(x) (threshold crossing), K =
0.5 … 5, on p2_late / p3_late. Rule fixed in the code before any number: per arm, K* = best F1 in the worse validation
group. **K* = 5σ for x** (its S10 setting is already its best), **4σ for ŷ** (3.5σ ties within 0.001 on p2_late, 4σ
better on p3_late). F1 on validation, x vs ŷ: x peaks at 5σ (0.731 / 0.767); ŷ 0.766 / 0.810 at 5σ, **0.836 / 0.867 at
4σ**, 0.836 / 0.860 at 3.5σ, then falls (precision drops faster than recall rises). Below 2σ ŷ still finds 89–95 % of
spikes with 400–560 false alarms / min; x at 2σ has ~17 000 / min (ŷ is ~0 between spikes, x is not).

The test groups were then read **once**, at 5σ and K* only. **Post-hoc operating point** (chosen after the S10 test
numbers were seen, on validation days only; report it as such, S10 stands as the pre-registered result):

| test, p2_final / p3_final / p4_unseen | threshold on x, 5σ | ŷ 5σ (S10) | **ŷ 4σ** |
|---|---|---|---|
| F1 | 0.746 / 0.716 / 0.776 | 0.790 / 0.755 / 0.799 | **0.849 / 0.826 / 0.820** |
| precision | 0.746 / 0.753 / 0.795 | 0.919 / 0.925 / 0.881 | 0.848 / 0.846 / 0.757 |
| recall | 0.745 / 0.682 / 0.758 | 0.693 / 0.638 / 0.731 | **0.850 / 0.807 / 0.895** |
| false alarms / min | 177 / 110 / 191 | 43 / 25 / 96 | 108 / 73 / 288 |
| Good units with recall ≥ 0.8 | 46 / 31 / 50 % | 40 / 29 / 46 % | **80 / 77 / 87 %** |
| Good units with recall ≥ 0.95 | 27 / 22 / 34 % | 32 / 22 / 37 % | **39 / 25 / 43 %** |
| recall median, 6–9σ units | 0.61 / 0.58 / 0.57 | 0.53 / 0.52 / 0.52 | **0.81 / 0.80 / 0.80** |
| recall median, 9–12σ units | 0.71 / 0.71 / 0.74 | 0.65 / 0.66 / 0.70 | **0.86 / 0.87 / 0.90** |
| recall median, ≥ 12σ units | 0.97–0.99 | 0.99–1.00 | 0.99–1.00 |

Reading: −5σ was too strict for ŷ. At 4σ the spike decoder beats threshold crossing on F1 **and** recall in every group,
the unseen patient included, and finds ≥ 80 % of the spikes of 77–87 % of the Good units (thresholding: 31–50 %). The
small units are where it gains (+0.2 recall). Cost: precision falls to 0.76–0.85, and on P4 false alarms rise above x's
(288 vs 191 / min); on P2 / P3 they stay below. "Keeps all Good units" still does not hold: 39–43 % reach recall
≥ 0.95. Every value is a 5-seed mean (`figures/data/threshold.json`).

**Use (16 Oct, the user's decision).** The poster shows ŷ at −4σ only, next to threshold
crossing on x at −5σ, with both thresholds stated in the Evaluation text and the Table 1 caption ("set on validation
days"); Table 2 no longer shows F1 ŷ (a −5σ number; −4σ was swept at 125 Hz only). The paper keeps S10 at −5σ as the
pre-registered result and reports −4σ as a post-hoc operating point.

**ŷ vs threshold crossing per NIX Good unit** (supervisor's second point; from the S10 JSONs, no new run). Recall,
median per SNR bin, 125 Hz, 5-seed mean (SD ≤ 0.005), p2_final / p3_final / p4_unseen; threshold = −5σ on bp(x):

| unit SNR | threshold on x | ŷ |
|---|---|---|
| 6–9σ | 0.61 / 0.58 / 0.57 | 0.53 / 0.52 / 0.52 |
| 9–12σ | 0.71 / 0.71 / 0.74 | 0.65 / 0.66 / 0.71 |
| 12–18σ | 0.985 / 0.994 / 0.989 | **0.992 / 0.999 / 0.996** |
| ≥ 18σ | 0.970 / 0.977 / 0.979 | **0.993 / 0.996 / 0.997** |
| % units recall ≥ 0.95 | 27 / 22 / 34 | 32 / 22 / 36 |

ŷ beats thresholding on F1 in every group (0.790 / 0.755 / 0.799 vs 0.746 / 0.716 / 0.776), precision 0.88–0.93 vs
0.75–0.80, false alarms ÷2–4, and keeps every clear unit (≥ 12σ) at 0.99–1.00, above thresholding. It does **not**
keep all Good units: on 6–12σ units (most of them) recall is 3–8 points below thresholding (stricter on borderline
crossings).

**All Good units (161 / 562 / 1 086 on 131 / 493 / 887 electrodes)**, threshold on x vs ŷ (5-seed mean): F1 per
electrode (median) 0.751 / 0.738 / 0.769 vs **0.798 / 0.764 / 0.785**, ŷ higher on **85 / 68 / 74 %** of electrodes;
Victor–Purpura 1 ms 0.340 / 0.326 / 0.327 vs **0.306 / 0.303 / 0.317**; units with recall ≥ 0.95 27 / 22 / 34 % vs
32 / 22 / 36 %; but recall ≥ 0.8 46 / 31 / 50 % vs 40 / 29 / 46 %. Claim: ŷ reproduces the Good spike trains better
(F1, false alarms), it does not keep more Good units.

**Clear Good units (≥ 12σ: 58 / 117 / 392 units)**, threshold on x vs ŷ (5-seed mean, SD ≤ 2.6 points):
recall ≥ 0.95 in 74 / 91 / 86 % vs **88 / 98 / 95 %** of units; recall ≥ 0.99 in 14 / 48 / 38 % vs **61 / 76 / 81 %**;
mean recall 0.957 / 0.981 / 0.972 vs **0.968 / 0.991 / 0.984**; ŷ ≥ threshold on 83 / 89 / 90 % of units. Sorting
left out: median accuracy higher on ŷ online, but fewer units ≥ 0.8 on p3 / p4 (61 / 210 vs 72 / 231).

**How ŷ is compared with the NIX (Aitor's) spikes, stated once.** Target y = m·bp(x): the spike band of the normalised
recording inside a Tukey window (α 0.25) around each NIX Good time, 0 elsewhere and on artifacts. Only NIX times are
used, never NIX waveforms. ŷ therefore keeps spike amplitude in units of σ; it is **not** a unit-height template. Spikes
on ŷ = crossings of **−5σ, σ = MAD of bp(x) on that electrode** (stored at cache time), 1 ms dead time, outside
artifacts: the same detector as on x. Matching: one to one, nearest first, ±0.5 ms; recall vs NIX Good, precision vs
NIX Good + Bad (S9); per unit, recall per Good unit binned by SNR; Victor–Purpura per electrode. The supervisor
suggested a threshold of 1 "because ŷ predicts normalised shapes and zeros": that would hold for unit-height templates,
not for this target. The underlying point stands, though: ŷ is ≈ 0 off-spike, so −5σ is stricter than it needs, and the
6–12σ Good units ŷ misses have troughs often shallower than 5σ. A lower ŷ threshold is **not measured**; it would have
to be chosen on p2_late / p3_late and read on test once, labelled post-hoc (S10).

## S12-A2: spike decoder on the frozen encoder (git `d3a264d`, jobs 47345593–98 train, 47345599 score)

A fresh spike decoder trained on the frozen S10 encoder (`INIT_CHECKPOINT=runs/s8-ref-s{1,2,3}/step_200000.pt
FREEZE_ENCODER=1`, 3 seeds × 50 k, `experiments/s12_arms.json`), scored once at 50 k. Compared with the S10 models it
starts from (s8-ref s1–s3, 200 k). Mean ± SD over 3 seeds.

| group | | F1 ŷ | P | R | false alarms / min | VP 1 ms | retention ŷ |
|---|---|---|---|---|---|---|---|
| p2_late | S12 | 0.765 ± 0.001 | 0.920 | 0.655 | 29.7 | 0.316 | 0.888 |
| p3_late | S12 | 0.809 ± 0.000 | 0.927 | 0.717 | 31.0 | 0.272 | 0.922 |
| p2_final | S12 | 0.788 ± 0.000 | 0.921 | 0.688 | 40.7 | 0.307 | 0.889 |
| | S10 | 0.789 ± 0.000 | 0.918 | 0.691 | 42.6 | 0.307 | 0.892 |
| p3_final | S12 | 0.754 ± 0.000 | 0.926 | 0.636 | 25.0 | 0.305 | 0.906 |
| | S10 | 0.754 ± 0.001 | 0.926 | 0.636 | 25.0 | 0.304 | 0.907 |
| p4_unseen | S12 | 0.799 ± 0.001 | 0.882 | 0.730 | 95.1 | 0.317 | 0.935 |
| | S10 | 0.799 ± 0.000 | 0.881 | 0.730 | 95.1 | 0.318 | 0.935 |

**No difference** (all gaps ≤ 0.003, below the T7 floor of 0.005 on F1 ŷ). Validation matches S8 too (0.766 / 0.810).
x̂ metrics and the probe are identical to S10, as they must be with a frozen encoder. Reading: the spike information is
in the tokens; ŷ does not depend on the encoder being trained for it.

## Cross-correlograms between electrodes (git `05165f8`, jobs 47342522 quick / 47342523 full, `figures/data/ccg.json`)

Per electrode pair (both ≥ 100 crossings on x): CCG of the −5σ crossings, 0.5 ms bins, ±25 ms; a peak = ≥ 5 Poisson SD
over the flanks (≥ 15 ms), within ±5 ms. Seed 1 per rate. "Kept" = of x's peaked pairs, the share also peaked on the
source; "new" = of the source's peaked pairs, the share not peaked on x; reference = x first half vs second half.

| p2_final / p3_final / p4_unseen | pairs with a peak on x | shape r (peak on x) | peaks kept | new peaks | same lag bin |
|---|---|---|---|---|---|
| x̂ 125 Hz | 120 / 340 / 1 062 | 0.84 / 0.87 / **0.98** | 0.76 / 0.38 / **0.86** | **0.86** / 0.13 / 0.02 | 0.91 / 0.85 / 0.96 |
| ŷ 125 Hz | | 0.72 / 0.73 / 0.97 | 0.48 / 0.11 / 0.82 | 0.02 / 0.29 / 0.02 | 0.79 / 0.57 / 0.97 |
| x̂ 62.5 Hz | | 0.76 / 0.84 / 0.97 | 0.79 / 0.37 / 0.84 | 0.97 / 0.11 / 0.04 | 0.81 / 0.90 / 0.98 |
| x̂ 31.3 Hz | | 0.65 / 0.76 / 0.96 | 0.75 / 0.20 / 0.79 | 0.97 / 0.16 / 0.17 | 0.54 / 0.81 / 0.98 |
| PCA F240 | | 0.82 / 0.44 / 0.77 | 0.61 / 0.05 / 0.51 | 0.88 / 0.25 / 0.02 | 0.86 / 0.22 / 0.97 |
| x half vs half | | 0.12 / 0.65 / 0.86 | 0.28 / 0.37 / 0.82 | 0.08 / 0.10 / 0.02 | 0.82 / 0.89 / 0.98 |

P4 (the only group with many peaked pairs): x̂ keeps the between-electrode timing as well as the recording reproduces
itself (0.86 vs 0.82 kept, same lag 0.96), PCA keeps half. P3: x's own peaks are not stable (half vs half 0.37), so
no reading. P2: x̂ **adds** peaks (663 peaked pairs vs 120 on x at 125 Hz, ~3 200 at 62.5 / 31.3 Hz); PCA adds them too (605,
88 % new), ŷ does not (58, 2 % new). So it is a property of reconstructing P2's signal at this budget, not of the codec
alone; cause not checked (candidate: crossings shared across electrodes, e.g. common hash).

## Code issues found by the runs

- T4 crashed after the timing: an empty pytest summary. Fixed in `faff9cb`.
- The isosplit sorting test failed in the container (acc 0.55). The synthetic spike had a flat
  2-sample trough, so noise picked a 1-sample alignment and isosplit split every unit in two. With a
  one-sample trough: 2 clusters, acc 1.0. Real spikes with flat troughs could split the same way;
  this is worth checking in the sorting results.

## Non-causal ceiling submitted (19 Oct, git `d3437fc`)

The same codec with `CAUSAL=0`: every conv padded on both sides, so each output sees 6 000 samples (200 ms) of past
and future; training windows carry the 200 ms after the loss span, with the same window starts as the causal runs.
Nothing tuned (s5-c8 settings, S8 recipe, K*ŷ = 4σ). `s8-noncausal-s1` (seed 1, data order 11, paired with
`s8-ref-s1`): smoke 47557273 (300 steps, debug queue), then 47557274 (24 h legs to 200 k). Scored once on the test
groups at 125 Hz; Δ against the five causal seeds, floor = their spread. Rules in the design notes.

## Wavelet top-K and spike-weighted PCA (scored 19–20 Oct; jobs 47493681, 47538639, 47547480)

Rules fixed before any number in the design notes. Same readouts, detector and σ from bp(x); 32 numbers per frame (wavelet:
32 coefficients, positions not counted, which favours it). Each baseline is deterministic: one run, no seeds. Floor =
SD of the five codec seeds in the same group (`eval/{wavelet*,pcaw-*}.json`).

**Wavelet family, validation only (47493681, git `c730832`).** Retention x̂ p2_late / p3_late: **sym4 0.880 / 0.902**,
db4 0.864 / 0.889, haar 0.842 / 0.862 → sym4 (`wavelet`).

**Wavelet sym4 on the test groups, once (47538639), against the codec x̂ (mean of 5 seeds).**
p2_final / p3_final / p4_unseen:

| | codec 125 Hz | wavelet 125 Hz | codec 62.5 Hz | wavelet 62.5 Hz |
|---|---|---|---|---|
| retention x̂ | **0.921 / 0.934 / 0.957** | 0.889 / 0.889 / 0.918 | **0.883 / 0.900 / 0.924** | 0.797 / 0.790 / 0.833 |
| % units retention ≥ 0.95 | **45 / 42 / 60** | 35 / 26 / 40 | **38 / 29 / 43** | 33 / 23 / 36 |
| F1 x̂ | **0.786 / 0.746 / 0.797** | 0.735 / 0.704 / 0.769 | **0.782 / 0.749 / 0.795** | 0.709 / 0.674 / 0.744 |
| false alarms / min x̂ | 71 / 51 / 127 | 122 / 74 / 140 | 53 / 33 / 100 | 83 / 48 / 96 |
| waveform r / amplitude ratio | **0.983–0.987** / 0.978–0.985 | 0.967–0.975 / 0.956–0.964 | **0.965–0.973** / 0.948–0.961 | 0.924–0.938 / 0.920–0.935 |
| LFP r (worst band) | 0.991 / 0.999 / 0.999 | **1.000 / 1.000 / 1.000** | 0.986 / 0.996 / 0.987 | **1.000 / 0.999 / 1.000** |
| MUA r | 0.995 / 0.868 / 0.923 | **1.000 / 0.939 / 0.977** | 0.995 / 0.574 / 0.809 | 0.999 / **0.810 / 0.913** |

Codec floors at 125 Hz: retention ≤ 0.003, F1 ≤ 0.002, waveform r 0.001, MUA r ≤ 0.005; at 62.5 Hz MUA r ≤ 0.035.

**Spike-weighted PCA, validation only (47547480).** Test once (pcaw-100, p2_final / p3_final / p4_unseen × 125 / 62.5 Hz): job 47709754, submitted 21 Oct, git `d3437fc`. Retention x̂ p2_late / p3_late: w = 3 0.458 / 0.475, w = 10
0.480 / 0.498, w = 30 0.482 / 0.501, **w = 100 0.502 / 0.522** (F1 0.527 / 0.568, waveform r 0.902 / 0.912, MUA r
0.965 / 0.889). The rule (best mean retention) picks **w = 100**, the edge of the grid; retention rises with w, but slowly.
Plain PCA (w = 1) has no retention on p2_late / p3_late (T5 gave F1 only, before the S9 precision rule), so the gain from
weighting on validation is not measured; on the test groups plain PCA keeps 0.47–0.53.

**Reading.**
- **The wavelet is by far the strongest non-learned baseline:** 0.89–0.92 retention at 125 Hz, against 0.47–0.55 for
  PCA and plain sampling. As 05b predicted, an adaptive code follows the spike wherever it is in the frame. "Far above
  linear compression" holds only against fixed codes.
- **The codec stays above it on every metric scored against NIX Good spikes, in every group, beyond the floor.** Not on label-free crossings (corrected 4 Nov): at 125 Hz the wavelet keeps more label-free crossings on P2 and P3 (0.789 / 0.800 vs 0.738 / 0.786, floor ≤ 0.006), the codec on P4 (0.862 vs 0.842); at 62.5 Hz no difference on P2 (0.647 vs 0.640, floor 0.012), codec higher on P3 and P4 (0.715 / 0.799 vs 0.659 / 0.715).
  - At 125 Hz: retention +0.03 to +0.05 (10–20 × the floor), F1 +0.03 to +0.05, waveform r +0.012 to +0.018, fewer
    false alarms, and 10–20 points more units at ≥ 0.95.
  - At 62.5 Hz the gap doubles: retention +0.09 to +0.11, waveform r +0.035 to +0.045. There, 32 coefficients per
    16 ms must cover the LFP as well as the spikes.
- **The wavelet keeps the slow bands better at both rates.**
  - MUA r: +0.005 / +0.071 / +0.054 at 125 Hz; at 62.5 Hz no difference on p2, then +0.236 / +0.104.
  - LFP r: 1.000 against 0.986–0.999.
  - This is the same pattern as against PCA: the codec wins on spikes, not on the slow bands.
- **Weighting PCA towards spike frames does not rescue a linear code.** Even at 100× it keeps about half the spikes
  (0.50–0.52 on validation), while the codec keeps 0.92–0.95 and the wavelet 0.88–0.90. The NIX labels alone do not
  explain the codec's gain.
- **Not run or not computed:**
  - plain PCA on p2_late / p3_late;
  - the strict wavelet (16 values + 16 positions, `--k 16`);
  - sorting on the wavelet was computed after all (`wavelet_F*_K32_*.json`): x̂ 0.53 / 0.53 / 0.55 at 125 Hz, below the
    codec's 0.60 / 0.59 / 0.60.

## Spike-weighted PCA on the test groups, once (21 Oct, job 47709754, git `d3437fc`)

pcaw-100 (w picked on validation), p2_final / p3_final / p4_unseen, `eval/pcaw-100_F*_K32_*.json`. Plain PCA from
`eval/pca_F*`, crossings from `eval/crossings/`; codec = mean of the 5 S10 seeds (table above).

| | plain PCA 125 Hz | **pcaw-100 125 Hz** | plain PCA 62.5 Hz | **pcaw-100 62.5 Hz** |
|---|---|---|---|---|
| retention x̂ | 0.531 / 0.472 / 0.525 | 0.573 / 0.520 / 0.573 | 0.096 / 0.065 / 0.078 | 0.147 / 0.102 / 0.119 |
| % units retention ≥ 0.95 | 27 / 15 / 27 | 27 / 15 / 27 | 2 / 2 / 3 | 4 / 2 / 3 |
| F1 x̂ | 0.566 / 0.489 / 0.567 | 0.594 / 0.525 / 0.601 | 0.140 / 0.091 / 0.117 | 0.204 / 0.135 / 0.170 |
| false alarms / min x̂ | 28 / 9 / 29 | 32 / 11 / 34 | 13 / 5 / 12 | 15 / 5 / 13 |
| waveform r / amplitude ratio | 0.899–0.908 / 0.78–0.79 | 0.909–0.918 / 0.80–0.82 | 0.625–0.647 / 0.39–0.40 | 0.662–0.684 / 0.46–0.47 |
| LFP r (worst band) | 1.000 | 1.000 | 1.000 | 1.000 |
| MUA r | 0.995 / 0.867 / 0.960 | 0.996 / 0.881 / 0.964 | 0.937 / 0.569 / 0.816 | 0.962 / 0.629 / 0.848 |
| crossings kept x̂ (label-free) | 0.34–0.40 | 0.416 / 0.380 / 0.448 | | 0.104 / 0.083 / 0.089 |
| sorting acc x̂ (median) | | 0.342 / 0.296 / 0.323 | | 0.048 / 0.035 / 0.040 |

Codec at 125 Hz for the same rows: retention 0.921 / 0.934 / 0.957, sorting x̂ 0.604 / 0.585 / 0.595 (x 0.552 / 0.564
/ 0.567), crossings 0.738 / 0.786 / 0.862.

**Reading.**
- **The test confirms validation: the labels do not rescue a linear code.** At 125 Hz, weighting adds +0.04 to +0.05
  retention over plain PCA (the codec floor is ≤ 0.003, so the gain is real), but pcaw-100 still keeps about half the spikes:
  0.35–0.41 below the codec in every group. The share of units at ≥ 0.95 hardly moves (+0.2 to +0.6 points).
- **At 62.5 Hz, both PCAs collapse** (retention 0.10–0.15, waveform r ≈ 0.67): 32 linear coefficients per 16 ms go
  to the slow bands, while the codec keeps 0.88–0.92.
- Sorting on pcaw-100 x̂ is far below sorting on x (0.30–0.34 vs 0.55–0.57); the codec's x̂ sorts as well as x or better.
- PCA keeps the LFP better at both rates (1.000). MUA depends on the rate. At 125 Hz pcaw-100 is no different on P2 and
  higher on P3 / P4 (0.881 / 0.964 vs 0.868 / 0.923). At 62.5 Hz the codec is higher on P2 (0.995 vs 0.962, floor 0.008),
  no difference on P3 (floor 0.082), pcaw-100 higher on P4 (0.848 vs 0.809, floor 0.018).
- Spike-weighted PCA is now complete on the rules of 05b (validation picked w; test read once).

## Non-causal ceiling: what real time costs (scored 27 Oct, read 4 Nov)

`s8-noncausal-s1` reached 200 k: leg 1 (47557274) hit its 24 h limit at 194.6 k, leg 2 (47558710) finished it on 22 Oct.
Scored once by `scripts.evaluate` on the test groups at 125 Hz (job 48009562, git `d3437fc`,
`eval/s10_s8-noncausal-s1_200000_<group>.json`); `compare --stage s10` is not used because it would rewrite `s10.json`.
Comparable with the S10 files (git `97b540a`): `metrics.py` and `evaluate.py` only gained metrics in between, and the
x-only scores (detect x, sorting x) are identical in every group. Rules in the design notes; nothing was tuned.

Δ = ceiling − mean of the 5 causal seeds, floor = their range; p2_final / p3_final / p4_unseen. ŷ at 5σ:

| | ceiling (1 seed) | causal (5 seeds) | Δ | floor |
|---|---|---|---|---|
| retention x̂ | 0.942 / 0.954 / 0.969 | 0.921 / 0.934 / 0.957 | **+0.022 / +0.020 / +0.012** | 0.006 / 0.006 / 0.004 |
| retention ŷ | 0.921 / 0.934 / 0.955 | 0.894 / 0.909 / 0.936 | **+0.027 / +0.025 / +0.018** | 0.007 / 0.008 / 0.004 |
| % units retention x̂ ≥ 0.95 | 61 / 72 / 86 | 45 / 42 / 60 | **+16 / +29 / +26** | 2.5 / 6.7 / 6.4 |
| F1 x̂ | 0.795 / 0.752 / 0.799 | 0.786 / 0.746 / 0.797 | +0.009 / +0.006 / +0.002 | 0.003 / 0.004 / 0.002 |
| F1 ŷ | 0.806 / 0.767 / 0.807 | 0.790 / 0.755 / 0.799 | +0.016 / +0.012 / +0.008 | 0.004 / 0.005 / 0.002 |
| waveform r | 0.996 / 0.997 / 0.997 | 0.983 / 0.985 / 0.987 | +0.013 / +0.012 / +0.009 | 0.003 / 0.003 / 0.002 |
| LFP r (worst band) | 0.995 / 0.999 / 0.999 | 0.991 / 0.999 / 0.999 | +0.004 / no difference / +0.001 | 0.003 / 0.000 / 0.001 |
| MUA r | 0.998 / 0.813 / 0.859 | 0.995 / 0.868 / 0.923 | no difference / **−0.055 / −0.065** | 0.003 / 0.008 / 0.014 |
| online sorting on ŷ | 0.629 / 0.593 / 0.586 | 0.603 / 0.571 / 0.573 | +0.026 / +0.022 / +0.013 | 0.018 / 0.011 / 0.007 |
| crossings kept x̂ (label-free) | 0.763 / 0.812 / 0.882 | 0.738 / 0.786 / 0.862 | +0.025 / +0.026 / +0.020 | 0.006 / 0.005 / 0.005 |

**Reading.**
- **Real time costs a little on average, more per unit.** Retention falls by 0.01–0.03 and waveform r by about 0.01, all
  beyond the floor. The share of units at ≥ 0.95 falls much more: 45 / 42 / 60 % causal against 61 / 72 / 86 %.
- **It is not a ceiling on every metric.** MUA r is worse on P3 and P4 with the look-ahead. Cause not checked.
- One seed: the floor is the causal seeds' range (05b rule), so the ceiling's own seed spread is unmeasured.
- Nothing about the causal codec changes (05b, S10).

**ŷ at 4σ (done 6 Nov).** The 05b rule sets K*ŷ = 4σ; the table above is at 5σ, the only threshold `evaluate.py` applies.
Job 48501320 (`logs/noncausal_ky4.py` on MN5, git `9b10c4c`, no repo code changed) read 4σ, and 5σ as a check, into
`eval/s10_s8-noncausal-s1_200000_ky4.json`. Ran 6 Nov after MN5 lifted the job limit (MaxJobs was 0 from 4 to 6 Nov).

| ŷ at 4σ, test | ceiling (1 seed) | causal (5 seeds) | Δ | floor |
|---|---|---|---|---|
| F1 | 0.866 / 0.836 / 0.825 | 0.849 / 0.826 / 0.820 | **+0.017 / +0.010 / +0.005** | 0.002 / 0.002 / 0.002 |
| precision | 0.862 / 0.849 / 0.755 | 0.848 / 0.846 / 0.757 | +0.014 / no difference / no difference | 0.004 / 0.006 / 0.005 |
| recall | 0.869 / 0.824 / 0.910 | 0.850 / 0.807 / 0.895 | **+0.019 / +0.017 / +0.015** | 0.004 / 0.005 / 0.004 |
| false alarms / min | 98 / 73 / 297 | 108 / 73 / 288 | −9 / no difference / +9 (floor 9.4) | 3.3 / 3.7 / 9.4 |
| Good units with recall ≥ 0.8 | 83 / 82 / 90 % | 80 / 77 / 87 % | +3.6 / +5.4 / +3.4 | 0.6 / 2.3 / 0.7 |
| Good units with recall ≥ 0.95 | 44 / 28 / 47 % | 39 / 25 / 43 % | +5.2 / +3.0 / +3.5 | 1.9 / 0.4 / 1.3 |
| recall median, 6–9σ units | 0.84 / 0.83 / 0.82 | 0.81 / 0.80 / 0.80 | +0.03 / +0.03 / +0.02 | 0.02 / 0.01 / 0.00 |
| recall median, 9–12σ units | 0.88 / 0.89 / 0.91 | 0.86 / 0.87 / 0.90 | +0.02 / +0.02 / +0.02 | 0.01 / 0.01 / 0.01 |

- At the 05b threshold (4σ), real time costs ŷ 0.005–0.017 F1 and 0.015–0.019 recall, beyond the floor in every group; precision differs only on P2. Smaller units lose the most (+0.02–0.03 recall median with the look-ahead).
- Check passed: the 5σ values in the same file match `evaluate.py` (F1 0.806 / 0.767 / 0.807).
- One seed; floor = range of the 5 causal seeds at 4σ (`figures/data/threshold.json`). The P4 false-alarm difference (+9.5) equals its floor (9.4): read as no clear difference.

## Latent space: what the 32 numbers hold (draft, 4 Nov, laptop, not committed)

`figure_data latent`, `experiments/figures/data/latent.{npz,json}`, figure `S_latent`.

**Draft: 2 of the 20 p4_unseen recordings, 120 s each, seed 1 (2 880 000 tokens). No paper claim from it yet.** Laptop CPU, 75 min, because MN5 is blocked. New analysis on a frozen model: nothing tuned, no S10 number changed.

| what is read from one token | each dimension alone (best \|ρ\|) | all 32, linear | all 32, nonlinear |
|---|---|---|---|
| spike in the token (AUC) | 0.08 | 0.64 | **0.98** |
| −5σ crossing, no labels (AUC) | 0.08 | 0.64 | **0.99** |
| spike time inside the 8 ms frame (R², spike tokens) | 0.17 | 0.20 | **0.77** |
| spike depth (R², spike tokens) | 0.04 | 0.02 | 0.65 |
| MUA, log power (R²) | 0.09 | 0.02 | 0.81 |
| slow voltage (R²) | **0.96** | **0.94** | 0.89 |
| signed LFP band signal, δ to high γ (R²) | 0.24–0.44 | 0.14–0.23 | 0.18–0.26 |
| LFP log band power, δ to γ (R²) | ≤ 0.04 | 0.00 | 0.05–0.19 |
| LFP high γ log power (R²) | 0.06 | 0.01 | 0.49 |

- **Slow voltage sits almost in one dimension** (the one with the largest variance, ρ 0.96). Spikes are spread over many dimensions and read only nonlinearly (AUC 0.98 against 0.64). This is the same finding as the linear probe.
- **Spike tokens separate from the rest**, on thin arcs along which the spike's time in the frame changes smoothly. Neighbours sharing the spike label (15 nearest, chance 0.50): 0.90 in the 32 dimensions, 0.91 on t-SNE, 0.70 on PC1–2. A second t-SNE with other tokens and seed agrees on the separation (0.90, 0.91, 0.69); the arcs are laid out differently, as expected.
- **Most of the 32 dimensions are used:** participation ratio 27.7 of 32.
- **Units do not form clean clusters** on the one electrode tested (3 units): same-unit neighbours 0.83 against 0.72 by chance from class sizes. One electrode, so this is an example, not a result.
- **One token does not hold LFP band power.** Reading signed bands improves with context, on one recording, 20 s: 1 token 0.20–0.26, 8 tokens 0.38–0.67, 48 tokens 0.51–0.66. This agrees with the decoder reading many tokens, and x̂ vs x stays at median r 0.9998 at 1 kHz on that recording. The first version of the panel only had log power and looked as if the LFP were missing; the voltage and signed-band rows were added after that check failed, and the power rows were kept.
- Not done: all test groups, more seeds (dimensions are not comparable between seeds).

## Linear baselines on the codec's training data (submitted 4 Nov, git `9b10c4c`)

Rules fixed before any number in the design notes. `pcafull` / `pcawfull-<w>`: PCA on every training frame (the old `pca_*`, `pcaw-100_*` used the first 2 s of each session and stay on disk). `lae-<rate>-s<seed>`: trained linear autoencoder, plain MSE, on the codec's recordings, window draws and optimiser, 5 seeds × 200 k at 125 / 62.5 / 31.3 Hz. Test groups are scored only after the autoencoder is within 5 % of `pcafull` on the validation windows. PCA sums: jobs 48505155–57.

## Memoria review: extra numbers from existing JSONs (12 Nov, laptop, not committed)

**Post hoc, nothing re-scored or re-tuned.** Asked by the TFG review. `python -m scripts.review_extra` →
`experiments/review_extra.json` (every number with its source JSON) and `experiments/review_tables.tex`. Mean ± SD
(ddof=1) over the 3 test groups; codec = 5-seed mean per group.

- **Wavelet sym4 at 125 Hz** (`eval/wavelet_F240_K32_*`): retention 0.899 ± 0.017, label-free crossings 0.810 ± 0.028,
  F1 0.736 ± 0.033, waveform r 0.970 ± 0.004, sorting 0.537 ± 0.008 (x in the same run 0.561), LFP worst band 1.000,
  MUA 0.972 ± 0.031.
- **bp(x̂) at −5σ** (S10 JSONs): P 0.863 ± 0.012, R 0.706 ± 0.046, F1 0.776 ± 0.027, 83 ± 39 false alarms / electrode-min.
- **Unit recall, 9–12σ** (median): x −5σ 0.718 ± 0.018, bp(x̂) 0.692 ± 0.026, ŷ −5σ 0.671 ± 0.030, ŷ −4σ 0.876 ± 0.019.
- **F1 of x̂ per group** (5-seed mean): p2_late 0.765, p3_late 0.800 (S9), p2_final 0.786, p3_final 0.746, p4_unseen 0.797.
- **False alarms are per electrode-minute** (electrodes with Good spikes). ŷ at −4σ: 108 / 73 / 288; x at −5σ: 177 / 110 / 191;
  ŷ at −5σ: 43 / 25 / 96. Detections on Bad spikes (not false by S9): ŷ −4σ 124 / 50 / 173. P4 Good units fire faster
  (mean 11.0 Hz vs 7.6 / 6.4; 808 Good spikes per electrode-min vs 561 / 435) at the same 1.2 units per electrode.
- **Rounding:** the memoria's mean ± SD were taken from group values already rounded to 3 decimals; from the exact values
  some cells move by 1 in the last digit (e.g. retention SD 0.019, not 0.018).
- **Submitted to MN5 12 Nov (jobs 48985507, 48985509–48985513 (evaluate), 48985514 (timing); cluster tree 9b10c4c, no deploy, `scripts/review_timing.py` copied untracked):** PCA / plain-sampling sorting and plain-sampling crossings (`scripts.evaluate … --out
  experiments/eval/review/…`), folded timing at 62.5 / 31.3 Hz and encoder only (`scripts.review_timing`).
