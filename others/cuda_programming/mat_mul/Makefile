CC = gcc
CXX = g++
NVCC = /usr/local/cuda/bin/nvcc
INCLUDE = -I../common/inc

mat_mul_cpu: mat_mul_cpu.cpp
	$(CXX) mat_mul_cpu.cpp -o mat_mul_cpu

mat_mul_cuda: mat_mul_cuda.cu
	$(NVCC) -ccbin g++ -m64 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80 -lcudart $(INCLUDE) mat_mul_cuda.cu -o mat_mul_cuda

clean:
	rm -f mat_mul_cpu mat_mul_cuda *.o
