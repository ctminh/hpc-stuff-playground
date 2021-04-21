# require a minimum version of cmake
cmake_minimum_required(VERSION 3.10)

# project name
project(tl_mxm_rpc_rdma)

# find MPI package
find_package(MPI REQUIRED COMPONENTS CXX)
message(STATUS "found mpi.h at ${MPI_CXX_INCLUDE_DIRS}")

# set mpicxx flags
set(MPI_FLAGS -lmpicxx -lpthread -lrt)

# compile the code
add_executable(tl_mxm tl_mxm_rdma_transf.cpp)
target_link_libraries(tl_mxm -lthallium -lmargo -lmercury -labt ${MPI_FLAGS})