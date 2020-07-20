# Flags for compiling
CFLAGS += $$(pkg-config --cflags starpu-1.2)
# LDLIBS += $$(pkg-config --libs starpu-1.3) -lrt -lcudart -L/usr/local/cuda/lib64 -lOpenCL -lstdc++ -lm
LDLIBS += $$(pkg-config --libs starpu-1.2) -lrt -lstdc++ -lm
CC = gcc
# NVCC = /usr/local/cuda/bin/nvcc

# for the test program
vector: vector_scal.c
	$(CC) $(CFLAGS) vector_scal.c -o vector $(LDLIBS)

clean:
	rm -fr *.o $() vector
