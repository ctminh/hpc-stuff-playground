# Flags for compiling
CFLAGS += $$(pkg-config --cflags starpu-1.2)
# LDLIBS += $$(pkg-config --libs starpu-1.3) -lrt -lcudart -L/usr/local/cuda/lib64 -lOpenCL -lstdc++ -lm
LDLIBS += $$(pkg-config --libs starpu-1.2) -lrt -lstdc++ -lm lcudart
CC = gcc
NVCC = /usr/local/cuda/bin/nvcc

vector_scal: vector_scal.o scal_cpu.o scal_cuda.o

# for the test program
%.o: %.cu
	$(NVCC) $(LDLIBS) $< -c $@

clean:
	rm -f vector_scal *.o
