CXX_COMPILER=g++
C_COMPILER=gcc
CUDA_COMPILER=nvcc
CFLAGS=-std=c++17 -O2
LDLIBS=-pthread


# for compiling
saxpy_cuda: saxpy_cudaflow.cu
	$(CUDA_COMPILER) $^ $(CFLAGS) -o $@

run:
	./saxpy_cuda

run_tfprof:
    TF_ENABLE_PROFILER=simple_test.json ./saxpy_cuda

clean:
    rm -f saxpy_cuda *.o
