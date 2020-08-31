CC = gcc
CXX = g++
NVCC = /usr/local/cuda/bin/nvcc

objects = main.o kernel.o

all: $(objects)
	$(CXX) $(objects) -o monte_carlo

main.o: main.cpp kernel.o dev_array.h
	$(CXX) -c $<

kernel.o: kernel.cu kernel.h
	$(NVCC) -c -m64 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80 -lcudart kernel.cu

clean:
	rm -f monte_carlo *.o
