# Some modules could be usefule or required to load first on BEAST
#   module load cuda/11.1.1
#   module load llvm/11.0.0_nvptx_offloading
#
#   libelf/0.8.13-gcc-8.3.1-knp36eg
#   llvm/15.0.4-gcc-8.3.1-m6h54vu
#   cuda/11.5.0-gcc-8.3.1-fvzhxgf
#   hwloc/2.8.0-gcc-8.3.1-sooe2gl
#   module load ncurses 
#
#   module load cuda/11.1.1
#   module load gcc_nvptx_offloading/10.2.0

ifeq ($(NODE),thx_nvcc) # Thx-NVCC
	CXX_FLAGS=-lm
	CXX=nvcc
else
	CXX=g++
	CXX_FLAGS=-std=c++11 -g
endif

# How to compile
#   NODE=<system_compiler> make

all: example1

example1: example1.cpp
	${CXX} ${CXX_FLAGS} $< -o $@

example1_cuda: example1.cu
	${CXX} ${CXX_FLAGS} example1.cu -o $@

clean:
	rm -f example1
