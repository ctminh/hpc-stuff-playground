# require a minimum version of cmake
cmake_minimum_required(VERSION 3.10)

# project name
project(tl_mxm_rpc_rdma_mpiomp)

# find MPI package
find_package(MPI REQUIRED COMPONENTS CXX)
message(STATUS "found mpi.h at ${MPI_CXX_INCLUDE_DIRS}")

# find OpenMP package
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(OPENMP_FLAGS OpenMP::OpenMP_CXX)
endif()

# set mpicxx flags
set(MPI_FLAGS -lpthread -lrt ${OPENMP_FLAGS}) # -lmpicxx

# compile the code
add_executable(tl_string_rdma_transf tl_string_rdma_transf.cpp)
target_link_libraries(tl_string_rdma_transf -lboost_serialization -lthallium -lmargo -lmercury -labt ${MPI_FLAGS})