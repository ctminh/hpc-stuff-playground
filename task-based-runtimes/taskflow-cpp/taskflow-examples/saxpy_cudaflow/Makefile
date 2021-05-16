CXX_COMPILER=g++
C_COMPILER=gcc
CUDA_COMP_SW=nvcc # for cuda11
CUDA_COMP_LC=/usr/local/cuda/bin/nvcc   # cuda10
CFLAGS=-std=c++17 -O2 -arch=compute_70 -code=sm_70 # arch of nvidia tesla v100
LDLIBS=-pthread


# for compiling
saxpy_cuda: saxpy_cudaflow.cu
	$(CUDA_COMP_SW) $^ $(CFLAGS) -o $@

run:
	./saxpy_cuda

run_tfprof:
	TF_ENABLE_PROFILER=cudaflow_test.json ./saxpy_cuda

clean:
	rm -f saxpy_cuda *.o