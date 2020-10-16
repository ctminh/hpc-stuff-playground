# Flags for compiling
ITAC_TRACE += -DTRACE -I$(VT_ROOT)/include -trace
CFLAGS += $$(pkg-config --cflags starpu-1.3 starpumpi-1.3)
LDLIBS += $$(pkg-config --libs starpu-1.3 starpumpi-1.3) -lrt -lcudart -L/usr/local/cuda/lib64 -lOpenCL -lstdc++ -lm $(ITAC_TRACE)
CUDA_FLAGS += -ccbin g++ -m64 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
CC = /home/ctminh/intel/compilers_and_libraries_2020.2.254/linux/mpi/intel64/bin/mpicc
MPICC = /home/ctminh/intel/compilers_and_libraries_2020.2.254/linux/mpi/intel64/bin/mpicc
NVCC = /usr/local/cuda/bin/nvcc

# declare object files
OBJECTS: stencil.o stencil-blocks.o stencil-tasks.o

# for compiling
stencil: $(OBJECTS)
	$(MPICC) $(CFLAGS) $(LDLIBS) -o stencil $(OBJECTS)
	
stencil.o: stencil.c stencil.h
	$(MPICC) $(CFLAGS) $(LDLIBS) -c stencil.c

stencil-blocks.o: stencil-blocks.c stencil.h
	$(MPICC) $(CFLAGS) $(LDLIBS) -c stencil-blocks.c

stencil-tasks.o: stencil-tasks.c stencil.h
	$(MPICC) $(CFLAGS) $(LDLIBS) -c stencil-tasks.c

clean:
	rm -f stencil *.o stencil.prot stencil.stf stencil.stf.*