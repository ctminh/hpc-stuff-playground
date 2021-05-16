# Flags for compiling
ITAC_TRACE += -DTRACE -I$(VT_ROOT)/include -trace
CFLAGS += $$(pkg-config --cflags starpu-1.3 starpumpi-1.3) $(ITAC_TRACE)
LDLIBS += $$(pkg-config --libs starpu-1.3 starpumpi-1.3) -lrt -lcudart -L/usr/local/cuda/lib64 -lOpenCL -lstdc++ -lm -L$(VT_LIB_DIR) -lVT $(VT_ADD_LIBS)
CUDA_FLAGS += -ccbin g++ -m64 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
CC = /home/ctminh/intel/compilers_and_libraries_2020.2.254/linux/mpi/intel64/bin/mpicc
MPICC = /home/ctminh/intel/compilers_and_libraries_2020.2.254/linux/mpi/intel64/bin/mpicc
NVCC = /usr/local/cuda/bin/nvcc

# declare object files
stencil: stencil.o stencil-blocks.o stencil-tasks.o stencil-kernels.o life.o

# for compiling
%.o: %.cu
	$(NVCC) $(CUDA_FLAGS) $(CFLAGS) $(LDLIBS) $< -c $@

clean:
	rm -f stencil *.o stencil.prot stencil.stf stencil.stf.*