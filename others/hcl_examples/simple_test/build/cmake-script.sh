rm -r CMakeCache.txt  ./CMakeFiles cmake_install.cmake  Makefile

# load cmake
module load cmake/3.14.4

# load gcc/9
module load gcc/9
export C_COMPILER=/lrz/sys/compilers/gcc/9.2/bin/gcc
export CXX_COMPILER=/lrz/sys/compilers/gcc/9.2/bin/g++

# export mpi-headers
module load intel
export INCLUDE=/lrz/sys/intel/impi2019u6/impi/2019.6.154/intel64/include:$INCLUDE
export CPATH=/lrz/sys/intel/impi2019u6/impi/2019.6.154/intel64/include:$CPATH
export MPI_CXX=/lrz/sys/intel/impi2019u6/impi/2019.6.154/intel64/lrzbin/mpicxx

# load rpclib
module use ~/.modules
module load rpclib

# load boost local
module load boost-1.72

# load hcl
module load hcl-dev

# run cmake
cmake -HCL_ENABLE_RPCLIB=true -DCMAKE_PREFIX_PATH=/dss/dsshome1/0A/di49mew/hcl/install/lib64 -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_MPI_CXX_COMPILER=${MPI_CXX} ..