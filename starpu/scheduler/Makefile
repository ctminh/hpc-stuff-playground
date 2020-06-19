# Flags for compiling
CFLAGS += $$(pkg-config --cflags starpu-1.3)
# LDLIBS += $$(pkg-config --libs starpu-1.3) -lrt -lcudart -L/usr/local/cuda/lib64 -lOpenCL -lstdc++ -lm
LDLIBS += $$(pkg-config --libs starpu-1.3) -lrt -lstdc++ -lm
CC = gcc
# NVCC = /usr/local/cuda/bin/nvcc

# for the test program
main: dummy_sched.c
	$(CC) $(CFLAGS) dummy_sched.c -o main $(LDLIBS)

clean:
	rm -fr *.o $() main
