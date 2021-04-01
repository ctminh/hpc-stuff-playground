# require a minimum version of cmake
cmake_minimum_required(VERSION 3.10)

# project name
project(margo_trans_bulkdat)

# compile the code
add_executable (server_bulkdata_sum server_bulkdata_sum.c)
target_link_libraries (server_sum -lmercury -lmargo -labt)

add_executable (client_call client_call.c)
target_link_libraries (client_call -lmercury -lmargo -labt)
