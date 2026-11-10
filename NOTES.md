# Study notes: what each kernel changes and why

Read this before putting the project on your resume. An interviewer will ask you to explain
any step, and to write kernel 3 on a whiteboard. Best practice: after reading a kernel, close
the file and rewrite it yourself from memory, then run the notebook to check it still PASSes.

Mental model: an NVIDIA GPU runs threads in **warps of 32**. A warp issues one instruction for
all 32 threads. Speed comes from (1) keeping memory accesses **coalesced** (a warp reading 32
consecutive floats becomes one 128-byte transaction), (2) **reusing** data from fast memory
(registers > shared memory > L2 > DRAM), and (3) keeping enough work in flight to hide latency.

SGEMM does 2·M·N·K flops on 3 matrices, so a good kernel is compute-bound. The whole game is
raising **arithmetic intensity**: flops performed per byte loaded from global memory.

## 1. Naive
One thread per output element. `row` comes from `threadIdx.x`, so the 32 threads in a warp
handle 32 different rows. Their reads of `A[row*K + k]` are K floats apart: 32 separate
memory transactions per load instead of 1. Expect only a few % of cuBLAS.

## 2. Coalesced
Swap the mapping so `threadIdx.x` picks the column. Now the warp reads 32 consecutive floats of
`B` (one transaction) and all threads read the same `A` element (a broadcast). Same math, often
5–10x faster. **Lesson:** on GPUs, the thread-to-data mapping matters as much as the algorithm.
(Same idea as your CPU loop-order exercise: i,k,j beat i,j,k ~12x.)

## 3. Shared-memory tiling
Each 32x32 block cooperatively loads a 32x32 tile of A and of B into `__shared__` memory, syncs,
then every thread reads 32 values from shared memory. Each global float is now reused 32 times.
`__syncthreads()` twice per tile: after loading (so nobody reads a half-loaded tile) and after
computing (so nobody overwrites a tile others still read). Bank conflicts: `As[ty][k]` is a
broadcast within a warp, `Bs[k][tx]` is 32 consecutive banks, so neither conflicts.

## 4. 1D block tiling
Now shared-memory bandwidth is the limit: 2 shared loads per FMA. Give each thread TM=8 outputs
in a column. It loads one `Bs` value into a register and uses it for 8 FMAs. Fewer threads
(512 per 64x64 tile), more work each. **Lesson:** register reuse is the next level after
shared-memory reuse.

## 5. 2D block tiling (register blocking)
Each thread computes an 8x8 tile as an outer product: load 8 values of A and 8 of B into
registers, do 64 FMAs. That is 64 FMAs per 16 shared loads (vs. 1 per 2 in kernel 3).
This is exactly your fast-matmul 8x8 NEON micro-kernel, moved to the GPU: the CPU version keeps
the 8x8 accumulator in NEON registers; here each thread keeps it in 64 scalar registers.
`__launch_bounds__(256)` tells the compiler the block size so it can budget registers.

## 6. Vectorized
Two changes on top of kernel 5:
- `float4` loads/stores move 128 bits per instruction instead of 32 (fewer instructions).
- A is stored **transposed** in shared memory (`As[k*BM + m]`), so the inner loop reads the 8
  values for `regM` contiguously, which the compiler can turn into vectorized shared loads.

## cuBLAS
NVIDIA's library. It is column-major, so `main.cu` computes `C^T = B^T A^T` to get a row-major
`C = A B` without any transposes (a classic interview question).

## Interview questions to practice
1. Why is kernel 1 slow? What does "coalesced" mean, in bytes and transactions?
2. Why two `__syncthreads()` per tile in kernel 3? What breaks if you remove each one?
3. What is arithmetic intensity, and how does each kernel raise it?
4. What limits occupancy for kernel 5? (registers per thread, shared memory per block)
5. Why not make the per-thread tile 16x16? (register pressure → spills to local memory)
6. What is still missing vs. cuBLAS? (double-buffering global→shared loads, warp tiling,
   Tensor Cores / TF32, autotuning tile sizes per GPU)
7. How is this similar to your CPU fast-matmul? (packing ≈ shared-memory staging; micro-kernel
   ≈ per-thread register tile; L1/L2 blocking ≈ block tiles)

## Next steps if you want more
- Kernel 7: warp tiling. Kernel 8: double-buffered shared memory (`cp.async` on sm_80+).
- Profile with Nsight Compute (`ncu`) and quote one metric per kernel (e.g. DRAM throughput,
  achieved occupancy). Interviewers love a profiler-backed explanation.
- Credit: the kernel progression follows Simon Boehm's widely used "How to Optimize a CUDA
  Matmul Kernel" worklog. Say so in the README; interviewers know it and respect honesty.
