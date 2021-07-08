# require a minimum version of cmake
cmake_minimum_required(VERSION 3.10)

# project name
project(hcl_queue_examples)

# set the example
set(examples queue_test,complextype_queue_test)

# fine MPI package
find_package(MPI REQUIRED COMPONENTS CXX)
message(STATUS "found mpi.h at ${MPI_CXX_INCLUDE_DIRS}")

# fine OpenMP package
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(DEFAULT_OPENMP_FLAG OpenMP::OpenMP_CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gdwarf-2 -g3")

option(HCL_ENABLE_RPCLIB "allow hcl to use RPCLIB" OFF)
option(HCL_ENABLE_THALLIUM_TCP "allow hcl to use THALLIUM_TCP" OFF)
option(HCL_ENABLE_THALLIUM_ROCE "allow hcl to use THALLIUM_ROCE" OFF)

if(HCL_ENABLE_RPCLIB)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHCL_ENABLE_RPCLIB")
    message("HCL_ENABLE_RPCLIB: ${HCL_ENABLE_RPCLIB}")
elseif(HCL_ENABLE_THALLIUM_TCP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHCL_ENABLE_THALLIUM_TCP")
    message("HCL_ENABLE_THALLIUM_TCP: ${HCL_ENABLE_THALLIUM_TCP}")
elseif(HCL_ENABLE_THALLIUM_ROCE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHCL_ENABLE_THALLIUM_ROCE")
    message("HCL_ENABLE_THALLIUM_ROCE: ${HCL_ENABLE_THALLIUM_ROCE}")
endif()

if(HCL_ENABLE_RPCLIB)
    set(RPC_LIB_FLAGS -lrpc ${RPC_LIB_FLAGS})
endif()

if(HCL_ENABLE_THALLIUM_ROCE OR HCL_ENABLE_THALLIUM_TCP)
    set(RPC_LIB_FLAGS -lthallium -lmercury -lmercury_util -lmargo -labt ${RPC_LIB_FLAGS})
endif()

set(LIB_FLAGS ${RPC_LIB_FLAGS} -lmpicxx -lpthread -lrt -lhcl ${DEFAULT_OPENMP_FLAG})

# Compile the example with linking hcl
foreach(exam ${examples})
    add_executable(${exam} ${exam}.cpp util.h)
    target_compile_options(${exam} PRIVATE -Wall -Wextra)
    target_link_libraries(${exam} "${LIB_FLAGS}") # MPI::MPI_CXX
endforeach()

# Another way to load hcl directly
#   because libhcl.so is linked by LD_PRELOAD, so don't need to link it in target_link_libs
#   set_target_properties(${example} PROPERTIES ENVIRONMENT LD_PRELOAD=/dss/dsshome1/0A/di49mew/hcl/install/lib64/libhcl.so)
