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
