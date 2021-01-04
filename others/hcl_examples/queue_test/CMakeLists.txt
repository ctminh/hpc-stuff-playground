# require a minimum version of cmake
cmake_minimum_required(VERSION 3.10)

# project name
project(hcl_example)

# set the example
set(example queue_test)

# fine MPI package
find_package(MPI REQUIRED COMPONENTS CXX)
message(STATUS "found mpi.h at ${MPI_CXX_INCLUDE_DIRS}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gdwarf-2 -g3")

option(HCL_ENABLE_RPCLIB "allow hcl to use RPCLIB" ON)

if(HCL_ENABLE_RPCLIB)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHCL_ENABLE_RPCLIB")
    message("HCL_ENABLE_RPCLIB: ${HCL_ENABLE_RPCLIB}")
endif()

if(HCL_ENABLE_RPCLIB)
    set(RPC_LIB_FLAGS -lrpc ${RPC_LIB_FLAGS})
endif()

# find HCL
find_library(HCL_LIB NAMES hcl PATHS /dss/dsshome1/0A/di49mew/hcl/install/lib64)
find_library(RPC_LIB NAMES rpc PATHS /dss/dsshome1/0A/di49mew/loc-libs/rpclib-master/install/lib)

# Compile the example
add_executable(${example} ${example}.cpp util.h)
# add_dependencies(${example} ${HCL_LIB})
set_target_properties(${example} PROPERTIES ENVIRONMENT LD_PRELOAD=/dss/dsshome1/0A/di49mew/hcl/install/lib64/libhcl.so)
target_compile_options(${example} PRIVATE -Wall -Wextra)
target_link_libraries(${example} "${HCL_LIB}" "${RPC_LIB}" MPI::MPI_CXX)
# target_link_libraries(${example} "${RPC_LIB}")
target_link_libraries(${example} ${RPC_LIB_FLAGS})
# set_target_properties (${example} PROPERTIES FOLDER ./)

message(INFO /dss/dsshome1/0A/di49mew/hcl/install/lib64/libhcl.so)

# Define MPI test case
# foreach (example ${examples})
#     mpi(ares 4 ${example} 2 500 1000 1 0)
# endforeach()