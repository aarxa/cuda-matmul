# cuda-matmul

Single-precision matrix multiply (SGEMM) on NVIDIA GPUs, from a naive kernel to a
register-blocked, vectorized one, checked and benchmarked against cuBLAS.
GPU companion to [fast-matmul](https://github.com/aarxa/fast-matmul) (the same problem on Apple M4 Pro with NEON).

| # | Kernel | Idea |
|---|---|---|
| 0 | cuBLAS | Reference and correctness oracle |
| 1 | Naive | One thread per output |
| 2 | Coalesced | Warp reads consecutive addresses |
| 3 | Shared-memory tiling | 32x32 tiles staged in `__shared__` |
| 4 | 1D block tiling | 8 outputs per thread, register reuse of B |
| 5 | 2D block tiling | 8x8 outer product per thread (register blocking) |
| 6 | Vectorized | `float4` global loads/stores, transposed A tile |

## Run it
No NVIDIA GPU locally? Open `colab.ipynb` in Google Colab, pick a T4 GPU runtime, Run all.

With a CUDA toolkit:
```bash
make            # ARCH=native by default; e.g. make ARCH=sm_75 for a T4 on older nvcc
./sgemm 1024 2048 4096
```
Every kernel is checked against cuBLAS before it is timed (PASS/FAIL column).

## Results
On a Tesla T4: naive 61.8 → vectorized 3,258 GFLOP/s at 4096 (~53x), **87.5% of cuBLAS**, every kernel verified against cuBLAS. Full table: [RESULTS.md](RESULTS.md).

## Credits
The kernel progression follows Simon Boehm's "How to Optimize a CUDA Matmul Kernel for
cuBLAS-like Performance" worklog. Study notes: [NOTES.md](NOTES.md).
