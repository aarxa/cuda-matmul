# ARCH=native targets the GPU in this machine (CUDA 11.6+). On older toolkits use e.g. ARCH=sm_75 (T4).
ARCH ?= native
NVCC ?= nvcc
FLAGS = -O3 -std=c++17 -arch=$(ARCH) -lineinfo

sgemm: src/main.cu src/kernels.cuh
	$(NVCC) $(FLAGS) -o $@ src/main.cu -lcublas

run: sgemm
	./sgemm

clean:
	rm -f sgemm

.PHONY: run clean
