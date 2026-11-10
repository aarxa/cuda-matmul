# Results

Google Colab, **Tesla T4** (Turing, sm_75, 40 SMs), CUDA 13.0, FP32, row-major square matrices.
Each kernel is checked against cuBLAS first (all PASS), then timed over 10 runs after 2 warm-ups.
Run 2026-11-09. The T4 is a 70 W card that throttles under load, so numbers vary ~10% between runs.

| Kernel | 1024 GFLOP/s | 2048 GFLOP/s | 4096 GFLOP/s | 4096 % of cuBLAS |
|---|---:|---:|---:|---:|
| cuBLAS | 4456.5 | 6933.8 | 3723.1 | 100.0% |
| 1 naive | 60.9 | 61.6 | 61.8 | 1.7% |
| 2 coalesced | 766.2 | 563.7 | 530.7 | 14.3% |
| 3 shared-mem tiling | 812.9 | 887.3 | 835.0 | 22.4% |
| 4 1D block tiling | 1727.8 | 1865.5 | 1698.3 | 45.6% |
| 5 2D block tiling | 3159.9 | 3769.7 | 2922.7 | 78.5% |
| 6 vectorized float4 | 3659.5 | 3946.6 | 3258.4 | 87.5% |

Naive → vectorized: **~53x at 4096** (61.8 → 3258 GFLOP/s), ~64x at 2048.

Correctness: max abs error vs cuBLAS was 8.0e-5 (1024) and 1.5e-4 (2048), and exactly 0 at 4096
(max |C| = 112). All six kernels accumulate the K products in the same order with FMA, so they
agree bit-for-bit with each other; cuBLAS matches that order at 4096 and uses a differently
ordered kernel at the smaller sizes.

Why cuBLAS pulls ahead at 2048: it picks a tuned kernel per shape (warp tiling, double-buffered
loads). Kernels 7–8 in NOTES.md ("next steps") close that gap.
