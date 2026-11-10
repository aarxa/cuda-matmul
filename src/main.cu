// Benchmark harness: checks every kernel against cuBLAS, then times it.
// Usage: ./sgemm [size ...]   (default sizes: 1024 2048 4096; each must be a multiple of 128)
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "kernels.cuh"

#define CUDA_CHECK(x)                                                                     \
  do {                                                                                    \
    cudaError_t err = (x);                                                                \
    if (err != cudaSuccess) {                                                             \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(1);                                                                            \
    }                                                                                     \
  } while (0)

#define CUBLAS_CHECK(x)                                                     \
  do {                                                                      \
    cublasStatus_t st = (x);                                                \
    if (st != CUBLAS_STATUS_SUCCESS) {                                      \
      fprintf(stderr, "cuBLAS error %d at %s:%d\n", st, __FILE__, __LINE__); \
      exit(1);                                                              \
    }                                                                       \
  } while (0)

constexpr int kWarmup = 2;
constexpr int kReps = 10;

static cublasHandle_t g_handle;

inline int cdiv(int a, int b) { return (a + b - 1) / b; }

// Launches kernel `id` once. Kernel 0 is cuBLAS.
void run(int id, int M, int N, int K, float alpha, const float *A, const float *B, float beta,
         float *C) {
  switch (id) {
    case 0:
      // cuBLAS is column-major. A row-major MxN matrix is a column-major NxM matrix, so
      // computing C^T = B^T A^T in column-major gives C = A B in row-major.
      CUBLAS_CHECK(cublasSgemm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K,
                               &beta, C, N));
      break;
    case 1:
      sgemm_naive<<<dim3(cdiv(M, 32), cdiv(N, 32)), dim3(32, 32)>>>(M, N, K, alpha, A, B, beta, C);
      break;
    case 2:
      sgemm_coalesced<<<dim3(cdiv(N, 32), cdiv(M, 32)), dim3(32, 32)>>>(M, N, K, alpha, A, B, beta, C);
      break;
    case 3:
      sgemm_smem<32><<<dim3(cdiv(N, 32), cdiv(M, 32)), dim3(32, 32)>>>(M, N, K, alpha, A, B, beta, C);
      break;
    case 4: {
      constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
      sgemm_1d_blocktile<BM, BN, BK, TM>
          <<<dim3(N / BN, M / BM), (BM * BN) / TM>>>(M, N, K, alpha, A, B, beta, C);
      break;
    }
    case 5: {
      constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
      sgemm_2d_blocktile<BM, BN, BK, TM, TN>
          <<<dim3(N / BN, M / BM), (BM * BN) / (TM * TN)>>>(M, N, K, alpha, A, B, beta, C);
      break;
    }
    case 6: {
      constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
      sgemm_vectorized<BM, BN, BK, TM, TN>
          <<<dim3(N / BN, M / BM), (BM * BN) / (TM * TN)>>>(M, N, K, alpha, A, B, beta, C);
      break;
    }
  }
}

const char *kNames[] = {"cuBLAS", "1 naive", "2 coalesced", "3 shared-mem tiling",
                        "4 1D block tiling", "5 2D block tiling", "6 vectorized float4"};
constexpr int kNumKernels = 7;

int main(int argc, char **argv) {
  std::vector<int> sizes;
  for (int i = 1; i < argc; ++i) sizes.push_back(atoi(argv[i]));
  if (sizes.empty()) sizes = {1024, 2048, 4096};

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s (sm_%d%d, %d SMs)\n\n", prop.name, prop.major, prop.minor,
         prop.multiProcessorCount);
  CUBLAS_CHECK(cublasCreate(&g_handle));

  const float alpha = 1.f, beta = 0.f;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  for (int n : sizes) {
    if (n % 128 != 0) {
      fprintf(stderr, "skipping %d: sizes must be multiples of 128\n", n);
      continue;
    }
    const int M = n, N = n, K = n;
    const size_t bytesA = sizeof(float) * M * K, bytesB = sizeof(float) * K * N,
                 bytesC = sizeof(float) * M * N;
    std::vector<float> hA(M * K), hB(K * N), hRef(M * N), hC(M * N);
    for (auto &x : hA) x = dist(rng);
    for (auto &x : hB) x = dist(rng);

    float *dA, *dB, *dC, *dRef;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));
    CUDA_CHECK(cudaMalloc(&dRef, bytesC));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(dRef, 0, bytesC));
    run(0, M, N, K, alpha, dA, dB, beta, dRef);
    CUDA_CHECK(cudaMemcpy(hRef.data(), dRef, bytesC, cudaMemcpyDeviceToHost));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double refMax = 0;
    for (float x : hRef) refMax = std::fmax(refMax, std::fabs(x));
    printf("M = N = K = %d   (max |C| from cuBLAS = %.3f, C[0] = %.5f)\n", n, refMax, hRef[0]);
    if (!(refMax > 0)) {
      fprintf(stderr, "reference result is all zeros or NaN; check is meaningless\n");
      exit(1);
    }
    printf("  %-22s %10s %10s %12s  %s\n", "kernel", "ms", "GFLOP/s", "% of cuBLAS", "check");
    double cublasGflops = 0;
    for (int id = 0; id < kNumKernels; ++id) {
      // Correctness: one run from a zeroed C, compared to cuBLAS.
      CUDA_CHECK(cudaMemset(dC, 0, bytesC));
      run(id, M, N, K, alpha, dA, dB, beta, dC);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost));
      double maxErr = 0;
      bool ok = true;
      for (int i = 0; i < M * N; ++i) {
        const double diff = std::fabs(hC[i] - hRef[i]);
        maxErr = std::fmax(maxErr, diff);
        if (!(diff <= 1e-2 + 1e-3 * std::fabs(hRef[i]))) ok = false;  // also catches NaN
      }

      // Timing: warm up, then average kReps back-to-back launches.
      for (int r = 0; r < kWarmup; ++r) run(id, M, N, K, alpha, dA, dB, beta, dC);
      CUDA_CHECK(cudaEventRecord(start));
      for (int r = 0; r < kReps; ++r) run(id, M, N, K, alpha, dA, dB, beta, dC);
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
      ms /= kReps;
      const double gflops = 2.0 * M * N * K / (ms * 1e6);
      if (id == 0) cublasGflops = gflops;
      printf("  %-22s %10.3f %10.1f %11.1f%%  %s (max err %.2e)\n", kNames[id], ms, gflops,
             100.0 * gflops / cublasGflops, ok ? "PASS" : "FAIL", maxErr);
    }
    printf("\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dRef));
  }
  CUBLAS_CHECK(cublasDestroy(g_handle));
  return 0;
}
