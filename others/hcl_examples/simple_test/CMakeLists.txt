# fine MPI package
find_package(MPI REQUIRED COMPONENTS CXX)
message(STATUS "found mpi.h at ${MPI_CXX_INCLUDE_DIRS}")

# project name
project(hcl_example)

# set the example
set (example simple_test)

# Compile the example
add_executable (${example} simple_test.cpp)
set_target_properties(${example} PROPERTIES ENVIRONMENT LD_PRELOAD=/dss/dsshome1/0A/di49mew/hcl/install/lib64/libhcl.so)
target_compile_options(${example} PRIVATE -Wall -Wextra)
target_link_libraries(${example} ${PROJECT_NAME} MPI::MPI_CXX)
# set_target_properties (${example} PROPERTIES FOLDER test)

message(INFO /dss/dsshome1/0A/di49mew/hcl/install/lib64/libhcl.so)

# Define MPI test case template
# function(mpi target mpi_procs example ranks_per_process num_requests size_of_request server_on_node debug)
#     set (test_parameters  -np ${mpi_procs} -f "${CMAKE_BINARY_DIR}/test/hostfile" "${CMAKE_BINARY_DIR}/test/${example}" ${ranks_per_process} ${num_requests} ${size_of_request} ${server_on_node} ${debug})
#     add_test(NAME ${target}_${example}_MPI_${mpi_procs}_${ranks_per_process}_${num_requests}_${size_of_request}_${server_on_node}_${debug} COMMAND "mpirun" ${test_parameters})
#     set_tests_properties(${target}_${example}_MPI_${mpi_procs}_${ranks_per_process}_${num_requests}_${size_of_request}_${server_on_node}_${debug} PROPERTIES ENVIRONMENT LD_PRELOAD=${CMAKE_BINARY_DIR}/libhcl.so)
# endfunction()

# Define MPI test case
# foreach (example ${examples})
#     mpi(ares 4 ${example} 2 500 1000 1 0)
# endforeach()