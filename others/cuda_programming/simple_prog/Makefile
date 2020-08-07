CC = gcc
CXX = g++
NVCC = /usr/local/cuda/bin/nvcc

simple_cpu: simple_cpu.cpp
	$(CXX) simple_cpu.cpp -o simple_cpu

simple_cuda: simple_cuda.cu
	$(NVCC) simple_cuda.cu -o simple_cuda

clean:
	rm -f simple_cpu simple_cuda *.o
