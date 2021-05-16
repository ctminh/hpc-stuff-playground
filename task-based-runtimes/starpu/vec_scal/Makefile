# Flags for compiling
CFLAGS += $$(pkg-config --cflags starpu-1.3)
LDLIBS += $$(pkg-config --libs starpu-1.3) -lrt -lcudart -L/usr/local/cuda/lib64 -lstdc++ -lm
CUDA_FLAGS += -ccbin g++ -m64 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
CC = gcc
NVCC = /usr/local/cuda/bin/nvcc

vector_scal: vector_scal.o scal_cpu.o scal_cuda.o

# for the test program
%.o: %.cu
	$(NVCC) $(CUDA_FLAGS) $(CFLAGS) $(LDLIBS) $< -c $@

clean:
	rm -f vector_scal *.o
