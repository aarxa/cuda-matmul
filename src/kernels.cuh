// SGEMM kernels, from naive to register-blocked and vectorized.
// All kernels compute C = alpha * A * B + beta * C for row-major A (MxK), B (KxN), C (MxN).
// Kernels 4-6 assume M, N, K are multiples of their block tile sizes (main.cu checks this).
#pragma once

// 1. Naive: one thread per output element.
//    threadIdx.x walks down rows, so neighbouring threads in a warp touch addresses K floats apart.
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A, const float *B,
                            float beta, float *C) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

// 2. Coalesced: same math, but threadIdx.x now walks along columns, so a warp reads
//    32 consecutive floats of B and writes 32 consecutive floats of C.
__global__ void sgemm_coalesced(int M, int N, int K, float alpha, const float *A, const float *B,
                                float beta, float *C) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

// 3. Shared-memory tiling: each block stages a TxT tile of A and of B in shared memory,
//    so every global load is reused T times instead of once.
template <int T>
__global__ void sgemm_smem(int M, int N, int K, float alpha, const float *A, const float *B,
                           float beta, float *C) {
  __shared__ float As[T][T];
  __shared__ float Bs[T][T];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row = blockIdx.y * T + ty;
  const int col = blockIdx.x * T + tx;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += T) {
    As[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + k0 + tx] : 0.f;
    Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(k0 + ty) * N + col] : 0.f;
    __syncthreads();
    for (int k = 0; k < T; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

// 4. 1D block tiling: each thread computes TM outputs down a column, so one value loaded
//    from Bs is reused TM times out of a register.
//    Launch with BM*BN/TM threads; needs BM*BK == BK*BN == thread count.
template <int BM, int BN, int BK, int TM>
__global__ void sgemm_1d_blocktile(int M, int N, int K, float alpha, const float *A,
                                   const float *B, float beta, float *C) {
  const int cRow = blockIdx.y, cCol = blockIdx.x;
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;
  const int innerColA = threadIdx.x % BK, innerRowA = threadIdx.x / BK;
  const int innerColB = threadIdx.x % BN, innerRowB = threadIdx.x / BN;
  float res[TM] = {0.f};
  for (int bk = 0; bk < K; bk += BK) {
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();
    A += BK;
    B += BK * N;
    for (int d = 0; d < BK; ++d) {
      const float b = Bs[d * BN + threadCol];
      for (int i = 0; i < TM; ++i) res[i] += As[(threadRow * TM + i) * BK + d] * b;
    }
    __syncthreads();
  }
  for (int i = 0; i < TM; ++i) {
    const int idx = (threadRow * TM + i) * N + threadCol;
    C[idx] = alpha * res[i] + beta * C[idx];
  }
}

// 5. 2D block tiling: each thread computes a TMxTN tile using an outer product of
//    TM values of A and TN values of B held in registers (TM*TN FMAs per TM+TN loads).
//    This is the GPU version of the 8x8 register-blocked micro-kernel in fast-matmul.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
    sgemm_2d_blocktile(int M, int N, int K, float alpha, const float *A, const float *B,
                       float beta, float *C) {
  constexpr int nThreads = (BM * BN) / (TM * TN);
  const int cRow = blockIdx.y, cCol = blockIdx.x;
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;
  const int innerRowA = threadIdx.x / BK, innerColA = threadIdx.x % BK;
  constexpr int strideA = nThreads / BK;
  const int innerRowB = threadIdx.x / BN, innerColB = threadIdx.x % BN;
  constexpr int strideB = nThreads / BN;
  float res[TM * TN] = {0.f};
  float regM[TM], regN[TN];
  for (int bk = 0; bk < K; bk += BK) {
    for (int o = 0; o < BM; o += strideA)
      As[(innerRowA + o) * BK + innerColA] = A[(innerRowA + o) * K + innerColA];
    for (int o = 0; o < BK; o += strideB)
      Bs[(innerRowB + o) * BN + innerColB] = B[(innerRowB + o) * N + innerColB];
    __syncthreads();
    A += BK;
    B += BK * N;
    for (int d = 0; d < BK; ++d) {
      for (int i = 0; i < TM; ++i) regM[i] = As[(threadRow * TM + i) * BK + d];
      for (int j = 0; j < TN; ++j) regN[j] = Bs[d * BN + threadCol * TN + j];
      for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) res[i * TN + j] += regM[i] * regN[j];
    }
    __syncthreads();
  }
  for (int i = 0; i < TM; ++i)
    for (int j = 0; j < TN; ++j) {
      const int idx = (threadRow * TM + i) * N + threadCol * TN + j;
      C[idx] = alpha * res[i * TN + j] + beta * C[idx];
    }
}

// 6. Vectorized: kernel 5 plus 128-bit (float4) global loads and stores, and A stored
//    transposed in shared memory so the inner loop reads As with unit stride.
//    Needs BM*BK/4 == BK*BN/4 == thread count (128x128x8 tile, 256 threads).
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
    sgemm_vectorized(int M, int N, int K, float alpha, const float *A, const float *B,
                     float beta, float *C) {
  const int cRow = blockIdx.y, cCol = blockIdx.x;
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);
  __shared__ float As[BK * BM];  // transposed: As[k * BM + m]
  __shared__ float Bs[BK * BN];
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;
  const int innerRowA = threadIdx.x / (BK / 4), innerColA = threadIdx.x % (BK / 4);
  const int innerRowB = threadIdx.x / (BN / 4), innerColB = threadIdx.x % (BN / 4);
  float res[TM * TN] = {0.f};
  float regM[TM], regN[TN];
  for (int bk = 0; bk < K; bk += BK) {
    const float4 a = reinterpret_cast<const float4 *>(&A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = a.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = a.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = a.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = a.w;
    reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4 *>(&B[innerRowB * N + innerColB * 4])[0];
    __syncthreads();
    A += BK;
    B += BK * N;
    for (int d = 0; d < BK; ++d) {
      for (int i = 0; i < TM; ++i) regM[i] = As[d * BM + threadRow * TM + i];
      for (int j = 0; j < TN; ++j) regN[j] = Bs[d * BN + threadCol * TN + j];
      for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) res[i * TN + j] += regM[i] * regN[j];
    }
    __syncthreads();
  }
  for (int i = 0; i < TM; ++i)
    for (int j = 0; j < TN; j += 4) {
      float4 *dst = reinterpret_cast<float4 *>(&C[(threadRow * TM + i) * N + threadCol * TN + j]);
      float4 c = dst[0];
      c.x = alpha * res[i * TN + j + 0] + beta * c.x;
      c.y = alpha * res[i * TN + j + 1] + beta * c.y;
      c.z = alpha * res[i * TN + j + 2] + beta * c.z;
      c.w = alpha * res[i * TN + j + 3] + beta * c.w;
      dst[0] = c;
    }
}
