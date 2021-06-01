# require a minimum version of cmake
cmake_minimum_required(VERSION 3.10)

# project name
project(mxm_task_stealing_prototype)

# find MPI lib
find_package(MPI REQUIRED COMPONENTS CXX)
message(STATUS "found mpi.h at ${MPI_CXX_INCLUDE_DIRS}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gdwarf-2 -g3")

option(HCL_ENABLE_RPCLIB "allow hcl to use RPCLIB" OFF)
option(HCL_ENABLE_THALLIUM_TCP "allow hcl to use THALLIUM_TCP" OFF)
option(HCL_ENABLE_THALLIUM_ROCE "allow hcl to use THALLIUM_ROCE" OFF)
option(ENABLE_ITAC_TRACE "allow the program to use Intel ITAC" OFF)
option(ENABLE_PARALLEL_OMP "allow the program to use OPENMP" OFF)
option(ENABLE_BOOST_STACKTRACE "allow the program to link boost-stacktrace" OFF)
option(ENABLE_GDB "allow debugging with gdb" OFF)
option(TEST_ON_LAPTOP "testing on the local laptop" OFF)

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

# the main flag for compiling the program
set(LIB_FLAGS ${RPC_LIB_FLAGS} -lmpicxx -lpthread -lrt -lhcl)

# enable ITAC for tracing
if(ENABLE_ITAC_TRACE)
    message("Intel_TRACE: ON")
    set(ITAC_FLAGS -DTRACE -I${VT_ROOT}/include -L${VT_LIB_DIR} -lVT ${VT_ADD_LIBS})
    set(LIB_FLAGS ${LIB_FLAGS} ${ITAC_FLAGS})
endif()

# enable Paralle with OpenMP threads in a single rank
if(ENABLE_PARALLEL_OMP)
    # find OpenMP lib
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        set(DEFAULT_OPENMP_FLAG OpenMP::OpenMP_CXX)
    endif()

    message("PARALLEL_OMP: ON")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPARALLEL_OMP")
    set(LIB_FLAGS ${LIB_FLAGS} ${DEFAULT_OPENMP_FLAG})
endif()

# enable linking the program with boost-stacktrace
if(ENABLE_BOOST_STACKTRACE)
    message("BOOST_STACKTRACE: ON")
endif()

# enable gbd debugging
if(ENABLE_GDB)
    message("ENABLE_GDB: ON")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb")
endif()

# Compile the example with linking hcl
add_executable(main main.cpp mxm_task_types.h mxm_kernel.h util.h)
target_compile_options(main PRIVATE -Wall -Wextra)
target_link_libraries(main "${LIB_FLAGS}") # MPI::MPI_CXX

