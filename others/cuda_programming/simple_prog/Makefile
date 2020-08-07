CC = gcc
NVCC = /usr/local/cuda/bin/nvcc

simple_cuda: simple_cuda.o

# for the test program
%.o: %.cu
	$(NVCC) $< -c $@

clean:
	rm -f simple_cuda *.o
